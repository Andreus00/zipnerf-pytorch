import os
import shutil

import torch
import glob

import logging


def restore_checkpoint(
        checkpoint_dir,
        model,
        optimizer
):
    dirs = glob.glob(os.path.join(checkpoint_dir, "*"))
    dirs.sort()
    path = dirs[-1] if len(dirs) > 0 else None
    if path is None:
        logging.info("Checkpoint does not exist. Starting a new training run.")
        init_step = 0
    else:
        logging.info("Restoring checkpoint from: {}".format(path))
        model.load_state_dict(torch.load(os.path.join(path, "model.pt")))
        if optimizer:
            optimizer.load_state_dict(torch.load(os.path.join(path, "optimizer.pt")))
        init_step = int(os.path.basename(path))
    return init_step, model, optimizer


def save_checkpoint(save_dir,
                    model, optimizer,
                    step=0,
                    total_limit=3):
    if total_limit > 0:
        folders = glob.glob(os.path.join(save_dir, "*"))
        folders.sort()
        for folder in folders[: len(folders) + 1 - total_limit]:
            shutil.rmtree(folder)
    # save model and optimizer
    save_path = os.path.join(save_dir, f"{step:06d}")
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
    logging.info("Checkpoint saved at: {}".format(save_path))
