import glob
import logging
import os
import shutil
import sys

import numpy as np
import random

import time

from absl import app
import gin
from internal import configs
from internal import datasets
from internal import image
from internal import models
from internal import train_utils
from internal import utils
from internal import vis
from internal import checkpoints
import torch
import tensorboardX
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from torch.utils._pytree import tree_map
import logging

configs.define_common_flags()

TIME_PRECISION = 1000  # Internally represent integer times in milliseconds.



def main(unused_argv):
    config = configs.load_config()
    config.exp_path = os.path.join("exp", config.exp_name)
    config.checkpoint_dir = os.path.join(config.exp_path, 'checkpoints')

    utils.makedirs(config.exp_path)
    with utils.open_file(os.path.join(config.exp_path, 'config.gin'), 'w') as f:
        f.write(gin.config_str())

    # setup logger
    logging.basicConfig(
        format="%(asctime)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(os.path.join(config.exp_path, 'log_train.txt'))],
        level=logging.INFO,
    )
    sys.excepthook = utils.handle_exception


    # setup model and optimizer
    model = models.Model(config=config)
    optimizer, lr_fn = train_utils.create_optimizer(config, model)

    # load dataset
    dataset = datasets.load_dataset('train', config.data_dir, config)
    test_dataset = datasets.load_dataset('test', config.data_dir, config)
    dataloader = torch.utils.data.DataLoader(np.arange(len(dataset)),
                                             num_workers=8,
                                             shuffle=True,
                                             batch_size=1,
                                             collate_fn=dataset.collate_fn,
                                             persistent_workers=True,
                                             )
    test_dataloader = torch.utils.data.DataLoader(np.arange(len(test_dataset)),
                                                  num_workers=4,
                                                  shuffle=False,
                                                  batch_size=1,
                                                  persistent_workers=True,
                                                  collate_fn=test_dataset.collate_fn,
                                                  )
    if config.rawnerf_mode:
        postprocess_fn = test_dataset.metadata['postprocess_fn']
    else:
        postprocess_fn = lambda z, _=None: z

    model = model.to(config.device)

    if config.resume_from_checkpoint:
        init_step, model, optimizer = checkpoints.restore_checkpoint(config.checkpoint_dir, model, optimizer)
    else:
        init_step = 0

    dataiter = iter(dataloader)
    test_dataiter = iter(test_dataloader)

    num_params = train_utils.tree_len(list(model.parameters()))
    logging.info(f'Number of parameters being optimized: {num_params}')

    if (dataset.size > model.num_glo_embeddings and model.num_glo_features > 0):
        raise ValueError(f'Number of glo embeddings {model.num_glo_embeddings} '
                         f'must be at least equal to number of train images '
                         f'{dataset.size}')

    # metric handler
    metric_harness = image.MetricHarness()

    # tensorboard
    summary_writer = tensorboardX.SummaryWriter(config.exp_path)
    # function to convert image for tensorboard
    tb_process_fn = lambda x: x.transpose(2, 0, 1) if len(x.shape) == 3 else x[None]

    if config.rawnerf_mode:
        for name, data in zip(['train', 'test'], [dataset, test_dataset]):
            # Log shutter speed metadata in TensorBoard for debug purposes.
            for key in ['exposure_idx', 'exposure_values', 'unique_shutters']:
                summary_writer.add_text(f'{name}_{key}', str(data.metadata[key]), 0)


    logging.info("Begin training...")
    step = init_step + 1
    total_time = 0
    total_steps = 0
    reset_stats = True
    if config.early_exit_steps is not None:
        num_steps = config.early_exit_steps
    else:
        num_steps = config.max_steps
    init_step = 0
    with logging_redirect_tqdm():
        tbar = tqdm(range(init_step + 1, num_steps + 1),
                    desc='Training', initial=init_step, total=num_steps)
        for step in tbar:
            try:
                batch = next(dataiter)
            except StopIteration:
                dataiter = iter(dataloader)
                batch = next(dataiter)

            batch = tree_map(lambda x: x.to(config.device) if x is not None else None, batch)
            if reset_stats:
                stats_buffer = []
                train_start_time = time.time()
                reset_stats = False

            # use lr_fn to control learning rate
            learning_rate = lr_fn(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            # fraction of training period
            train_frac = np.clip((step - 1) / (config.max_steps - 1), 0, 1)

            # Indicates whether we need to compute output normal or depth maps in 2D.
            compute_extras = (config.compute_disp_metrics or config.compute_normal_metrics)
            optimizer.zero_grad()

            renderings, ray_history = model(
                True,
                batch,
                train_frac=train_frac,
                compute_extras=compute_extras,
                zero_glo=False)

            losses = {}

            # supervised by data
            data_loss, stats = train_utils.compute_data_loss(batch, renderings, config)
            losses['data'] = data_loss

            # interlevel loss in MipNeRF360
            if config.interlevel_loss_mult > 0 and not model.single_mlp:
                losses['interlevel'] = train_utils.interlevel_loss(ray_history, config)

            # interlevel loss in ZipNeRF360
            if config.anti_interlevel_loss_mult > 0 and not model.single_mlp:
                losses['anti_interlevel'] = train_utils.anti_interlevel_loss(ray_history, config)

            # distortion loss
            if config.distortion_loss_mult > 0:
                losses['distortion'] = train_utils.distortion_loss(ray_history, config)

            # opacity loss
            if config.opacity_loss_mult > 0:
                losses['opacity'] = train_utils.opacity_loss(renderings, config)

            # orientation loss in RefNeRF
            if (config.orientation_coarse_loss_mult > 0 or
                    config.orientation_loss_mult > 0):
                losses['orientation'] = train_utils.orientation_loss(batch, model, ray_history,
                                                                     config)
            # hash grid l2 weight decay
            if config.hash_decay_mults > 0:
                losses['hash_decay'] = train_utils.hash_decay_loss(ray_history, config)

            # normal supervision loss in RefNeRF
            if (config.predicted_normal_coarse_loss_mult > 0 or
                    config.predicted_normal_loss_mult > 0):
                losses['predicted_normals'] = train_utils.predicted_normal_loss(
                    model, ray_history, config)
            loss = sum(losses.values())
            stats['loss'] = loss.item()
            stats['losses'] = tree_map(lambda x: x.item(), losses)

            loss.backward()
            # clip gradient by max/norm/nan
            train_utils.clip_gradients(model, config)
            optimizer.step()

            stats['psnrs'] = image.mse_to_psnr(stats['mses'])
            stats['psnr'] = stats['psnrs'][-1]

            # Log training summaries. This is put behind a host_id check because in
            # multi-host evaluation, all hosts need to run inference even though we
            # only use host 0 to record results.
            stats_buffer.append(stats)
            if step == init_step + 1 or step % config.logging.info_every == 0:
                elapsed_time = time.time() - train_start_time
                steps_per_sec = config.logging.info_every / elapsed_time
                rays_per_sec = config.batch_size * steps_per_sec

                # A robust approximation of total training time, in case of pre-emption.
                total_time += int(round(TIME_PRECISION * elapsed_time))
                total_steps += config.logging.info_every
                approx_total_time = int(round(step * total_time / total_steps))

                # Transpose and stack stats_buffer along axis 0.
                fs = [utils.flatten_dict(s, sep='/') for s in stats_buffer]
                stats_stacked = {k: np.stack([f[k] for f in fs]) for k in fs[0].keys()}

                # Split every statistic that isn't a vector into a set of statistics.
                stats_split = {}
                for k, v in stats_stacked.items():
                    if v.ndim not in [1, 2] and v.shape[0] != len(stats_buffer):
                        raise ValueError('statistics must be of size [n], or [n, k].')
                    if v.ndim == 1:
                        stats_split[k] = v
                    elif v.ndim == 2:
                        for i, vi in enumerate(tuple(v.T)):
                            stats_split[f'{k}/{i}'] = vi

                # Summarize the entire histogram of each statistic.
                for k, v in stats_split.items():
                    summary_writer.add_histogram('train_' + k, v, step)

                # Take the mean and max of each statistic since the last summary.
                avg_stats = {k: np.mean(v) for k, v in stats_split.items()}
                max_stats = {k: np.max(v) for k, v in stats_split.items()}

                summ_fn = lambda s, v: summary_writer.add_scalar(s, v, step)  # pylint:disable=cell-var-from-loop

                # Summarize the mean and max of each statistic.
                for k, v in avg_stats.items():
                    summ_fn(f'train_avg_{k}', v)
                for k, v in max_stats.items():
                    summ_fn(f'train_max_{k}', v)

                summ_fn('train_num_params', num_params)
                summ_fn('train_learning_rate', learning_rate)
                summ_fn('train_steps_per_sec', steps_per_sec)
                summ_fn('train_rays_per_sec', rays_per_sec)

                summary_writer.add_scalar('train_avg_psnr_timed', avg_stats['psnr'],
                                            total_time // TIME_PRECISION)
                summary_writer.add_scalar('train_avg_psnr_timed_approx', avg_stats['psnr'],
                                            approx_total_time // TIME_PRECISION)

                if dataset.metadata is not None and model.learned_exposure_scaling:
                    scalings = model.exposure_scaling_offsets.weight
                    num_shutter_speeds = dataset.metadata['unique_shutters'].shape[0]
                    for i_s in range(num_shutter_speeds):
                        for j_s, value in enumerate(scalings[i_s]):
                            summary_name = f'exposure/scaling_{i_s}_{j_s}'
                            summary_writer.add_scalar(summary_name, value, step)

                precision = int(np.ceil(np.log10(config.max_steps))) + 1
                avg_loss = avg_stats['loss']
                avg_psnr = avg_stats['psnr']
                str_losses = {  # Grab each "losses_{x}" field and logging.info it as "x[:4]".
                    k[7:11]: (f'{v:0.5f}' if 1e-4 <= v < 10 else f'{v:0.1e}')
                    for k, v in avg_stats.items()
                    if k.startswith('losses/')
                }
                logging.info(f'{step}' + f'/{config.max_steps:d}:' +
                            f'loss={avg_loss:0.5f},' + f'psnr={avg_psnr:.3f},' +
                            f'lr={learning_rate:0.2e} | ' +
                            ','.join([f'{k}={s}' for k, s in str_losses.items()]) +
                            f',{rays_per_sec:0.0f} r/s')  

                # Reset everything we are tracking between summarizations.
                reset_stats = True

            if step > 0 and step % config.checkpoint_every == 0:
                checkpoints.save_checkpoint(config.checkpoint_dir,
                                            model, optimizer, step,
                                            config.checkpoints_total_limit)

            # Test-set evaluation.
            if config.train_render_every > 0 and step % config.train_render_every == 0:
                # We reuse the same random number generator from the optimization step
                # here on purpose so that the visualization matches what happened in
                # training.
                eval_start_time = time.time()
                try:
                    test_batch = next(test_dataiter)
                except StopIteration:
                    test_dataiter = iter(test_dataloader)
                    test_batch = next(test_dataiter)

                test_batch = tree_map(lambda x: x.to(config.device) if x is not None else None, test_batch)
                # render a single image with all distributed processes
                rendering = models.render_image(model,
                                                test_batch, False,
                                                train_frac, config)

                # move to numpy
                rendering = tree_map(lambda x: x.detach().cpu().numpy(), rendering)
                test_batch = tree_map(lambda x: x.detach().cpu().numpy() if x is not None else None, test_batch)
                # Log eval summaries on host 0.
                
                eval_time = time.time() - eval_start_time
                num_rays = np.prod(test_batch['directions'].shape[:-1])
                rays_per_sec = num_rays / eval_time
                summary_writer.add_scalar('test_rays_per_sec', rays_per_sec, step)

                metric_start_time = time.time()
                metric = metric_harness(
                    postprocess_fn(rendering['rgb']), postprocess_fn(test_batch['rgb']))
                logging.info(f'Eval {step}: {eval_time:0.3f}s, {rays_per_sec:0.0f} rays/sec')
                logging.info(f'Metrics computed in {(time.time() - metric_start_time):0.3f}s')
                for name, val in metric.items():
                    if not np.isnan(val):
                        logging.info(f'{name} = {val:.4f}')  
                        summary_writer.add_scalar('train_metrics/' + name, val, step)

                if config.vis_decimate > 1:
                    d = config.vis_decimate
                    decimate_fn = lambda x, d=d: None if x is None else x[::d, ::d]
                else:
                    decimate_fn = lambda x: x
                rendering = tree_map(decimate_fn, rendering)
                test_batch = tree_map(decimate_fn, test_batch)
                vis_start_time = time.time()
                vis_suite = vis.visualize_suite(rendering, test_batch)
                with tqdm.external_write_mode():
                    logging.info(f'Visualized in {(time.time() - vis_start_time):0.3f}s')
                if config.rawnerf_mode:
                    # Unprocess raw output.
                    vis_suite['color_raw'] = rendering['rgb']
                    # Autoexposed colors.
                    vis_suite['color_auto'] = postprocess_fn(rendering['rgb'], None)
                    summary_writer.add_image('test_true_auto',
                                                tb_process_fn(postprocess_fn(test_batch['rgb'], None)), step)
                    # Exposure sweep colors.
                    exposures = test_dataset.metadata['exposure_levels']
                    for p, x in list(exposures.items()):
                        vis_suite[f'color/{p}'] = postprocess_fn(rendering['rgb'], x)
                        summary_writer.add_image(f'test_true_color/{p}',
                                                    tb_process_fn(postprocess_fn(test_batch['rgb'], x)), step)
                summary_writer.add_image('test_true_color', tb_process_fn(test_batch['rgb']), step)
                if config.compute_normal_metrics:
                    summary_writer.add_image('test_true_normals',
                                                tb_process_fn(test_batch['normals']) / 2. + 0.5, step)
                for k, v in vis_suite.items():
                    summary_writer.add_image('test_output_' + k, tb_process_fn(v), step)

    if config.max_steps > init_step:
        logging.info('Saving last checkpoint at step {} to {}'.format(step, config.checkpoint_dir))  
        checkpoints.save_checkpoint(config.checkpoint_dir,
                                    model, optimizer, step,
                                    config.checkpoints_total_limit)
    logging.info('Finish training.') 


if __name__ == '__main__':
    with gin.config_scope('train'):
        app.run(main)

