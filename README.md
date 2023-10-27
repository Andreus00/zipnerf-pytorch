# ZipNeRF
## Notes on this fork:
I created this fork to remove Accelerate as I had to integrate zipnerf in a system that is not written with Accelerate.
I hope that someone will find this repo useful as well.
I didn't check whether this implementation returns the same exact results as the original one, but I don't see why it should not since it
is the same code.

# ZipNeRF-pytorch's results and info
An unofficial pytorch implementation of 
"Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields" 
[https://arxiv.org/abs/2304.06706](https://arxiv.org/abs/2304.06706).
This work is based on [multinerf](https://github.com/google-research/multinerf), so features in refnerf,rawnerf,mipnerf360 are also available.

## News
- (6.22) Add extracting mesh through tsdf; add [gradient scaling](https://gradient-scaling.github.io/) for near plane floaters.
- (5.26) Implement the latest version of ZipNeRF [https://arxiv.org/abs/2304.06706](https://arxiv.org/abs/2304.06706).
- (5.22) Add extracting mesh; add logging,checkpointing system

## Results
New results(5.27): [Pretrained weights](https://drive.google.com/drive/folders/1W1jFa519m7Ye9Pcz5N_30TMPM-7KTTBc?usp=sharing)

360_v2:

https://github.com/SuLvXiangXin/zipnerf-pytorch/assets/83005605/2b276e48-2dc4-4508-8441-e90ec963f7d9


360_v2_glo:(fewer floaters, but worse metric)


https://github.com/SuLvXiangXin/zipnerf-pytorch/assets/83005605/bddb5610-2a4f-4981-8e17-71326a24d291






mesh results(5.27):

![mesh](https://github.com/SuLvXiangXin/zipnerf-pytorch/assets/83005605/35866fa7-fe6a-44fe-9590-05d594bdb8cd)



Mipnerf360(PSNR):

|           | bicycle | garden | stump | room  | counter | kitchen | bonsai |
|:---------:|:-------:|:------:|:-----:|:-----:|:-------:|:-------:|:------:|
|   Paper   |  25.80  | 28.20  | 27.55 | 32.65 |  29.38  |  32.50  | 34.46  |
| This repo |  25.44  | 27.98  | 26.75 | 32.13 |  29.10  |  32.63  | 34.20  |


Blender(PSNR):

|           | chair | drums | ficus | hotdog | lego  | materials |  mic  | ship  |
|:---------:|:-----:|:-----:|:-----:|:------:|:-----:|:---------:|:-----:|:-----:|
|   Paper   | 34.84 | 25.84 | 33.90 | 37.14  | 34.84 |   31.66   | 35.15 | 31.38 |
| This repo | 35.26 | 25.51 | 32.66 | 36.56  | 35.04 |   29.43   | 34.93 | 31.38 |

For Mipnerf360 dataset, the model is trained with a downsample factor of 4 for outdoor scene and 2 for indoor scene(same as in paper).
Training speed is about 1.5x slower than paper(1.5 hours on 8 A6000).

The hash decay loss seems to have little effect(?), as many floaters can be found in the final results in both experiments (especially in Blender).

## Install

```
# Clone the repo.
git clone https://github.com/SuLvXiangXin/zipnerf-pytorch.git
cd zipnerf-pytorch

# Make a conda environment.
conda create --name zipnerf python=3.9
conda activate zipnerf

# Install requirements.
pip install -r requirements.txt

# Install other extensions
## IMPORTANT: if yoi are on Linuz and gridencoder is saying that you are not using c++17, 
## try going in ./gridencoder/setup.py and changing line 13 to `c_flags = ['-O3', '-std=c++17']`.
pip install ./gridencoder

# Install nvdiffrast (optional, for textured mesh)
git clone https://github.com/NVlabs/nvdiffrast
pip install ./nvdiffrast

# Install a specific cuda version of torch_scatter 
# see more detail at https://github.com/rusty1s/pytorch_scatter
CUDA=cu117
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+${CUDA}.html
```

## Dataset
[mipnerf360](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip)

[refnerf](https://storage.googleapis.com/gresearch/refraw360/ref.zip)

[nerf_synthetic](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)

[nerf_llff_data](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)

```
mkdir data
cd data

# e.g. mipnerf360 data
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip 360_v2.zip
```

## Train
```

# Where your data is 
DATA_DIR=data/360_v2/bicycle
EXP_NAME=360_v2/bicycle

# Experiment will be conducted under "exp/${EXP_NAME}" folder
# "--gin_configs=configs/360.gin" can be seen as a default config 
# and you can add specific config useing --gin_bindings="..." 
python3 train.py \
    --gin_configs=configs/360.gin \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${EXP_NAME}'" \
    --gin_bindings="Config.factor = 4"

# or you can also run without accelerate (without DDP)
CUDA_VISIBLE_DEVICES=0 python train.py \
    --gin_configs=configs/360.gin \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${EXP_NAME}'" \
      --gin_bindings="Config.factor = 4" 

# alternatively you can use an example training script 
bash scripts/train_360.sh

# blender dataset
bash scripts/train_blender.sh

# metric, render image, etc can be viewed through tensorboard
tensorboard --logdir "exp/${EXP_NAME}"

```

### Render
Rendering results can be found in the directory `exp/${EXP_NAME}/render`
```
python3 render.py \
    --gin_configs=configs/360.gin \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${EXP_NAME}'" \
    --gin_bindings="Config.render_path = True" \
    --gin_bindings="Config.render_path_frames = 480" \
    --gin_bindings="Config.render_video_fps = 60" \
    --gin_bindings="Config.factor = 4"  

# alternatively you can use an example rendering script 
bash scripts/render_360.sh
```
## Evaluate
Evaluating results can be found in the directory `exp/${EXP_NAME}/test_preds`
```
# using the same exp_name as in training
python3 eval.py \
    --gin_configs=configs/360.gin \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${EXP_NAME}'" \
    --gin_bindings="Config.factor = 4"


# alternatively you can use an example evaluating script 
bash scripts/eval_360.sh
```

## Extract mesh
Mesh results can be found in the directory `exp/${EXP_NAME}/mesh`
```
# more configuration can be found in internal/configs.py
python3 extract.py \
    --gin_configs=configs/360.gin \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${EXP_NAME}'" \
    --gin_bindings="Config.factor = 4"
#    --gin_bindings="Config.mesh_radius = 1"  # (optional) smaller for more details e.g. 0.2 in bicycle scene
#    --gin_bindings="Config.isosurface_threshold = 20"  # (optional) empirical value
#    --gin_bindings="Config.mesh_voxels=134217728"  # (optional) number of voxels used to extract mesh, e.g. 134217728 equals to 512**3 . Smaller values may solve OutoFMemoryError
#    --gin_bindings="Config.vertex_color = True"  # (optional) saving mesh with vertex color instead of atlas which is much slower but with more details.
#    --gin_bindings="Config.vertex_projection = True"  # (optional) use projection for vertex color

# or extracting mesh using tsdf method
python3 tsdf.py \
    --gin_configs=configs/360.gin \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${EXP_NAME}'" \
    --gin_bindings="Config.factor = 4"

# alternatively you can use an example script 
bash scripts/extract_360.sh
```

## OutOfMemory
you can decrease the total batch size by 
adding e.g.  `--gin_bindings="Config.batch_size = 8192" `, 
or decrease the test chunk size by adding e.g.  `--gin_bindings="Config.render_chunk_size = 8192" `,
or use more GPU by configure `accelerate config` .


## Preparing custom data
More details can be found at https://github.com/google-research/multinerf
```
DATA_DIR=my_dataset_dir
bash scripts/local_colmap_and_resize.sh ${DATA_DIR}
```

## TODO
- [] Fix evaluate_color_projection in extract.py
- [x] Rewrite scripts in ./scripts so that they start files in python instead of passing through accelerate
- [x] Add a logger
- [x] Fix/Check/Remove all TODOs
- [x] Convert tsdf.py
- [x] Remove config.global_rank from the code
- [x] Remove config.world_size from the code

## Citation
```
@misc{barron2023zipnerf,
      title={Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields}, 
      author={Jonathan T. Barron and Ben Mildenhall and Dor Verbin and Pratul P. Srinivasan and Peter Hedman},
      year={2023},
      eprint={2304.06706},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{multinerf2022,
      title={{MultiNeRF}: {A} {Code} {Release} for {Mip-NeRF} 360, {Ref-NeRF}, and {RawNeRF}},
      author={Ben Mildenhall and Dor Verbin and Pratul P. Srinivasan and Peter Hedman and Ricardo Martin-Brualla and Jonathan T. Barron},
      year={2022},
      url={https://github.com/google-research/multinerf},
}

@Misc{accelerate,
  title =        {Accelerate: Training and inference at scale made simple, efficient and adaptable.},
  author =       {Sylvain Gugger, Lysandre Debut, Thomas Wolf, Philipp Schmid, Zachary Mueller, Sourab Mangrulkar},
  howpublished = {\url{https://github.com/huggingface/accelerate}},
  year =         {2022}
}

@misc{torch-ngp,
    Author = {Jiaxiang Tang},
    Year = {2022},
    Note = {https://github.com/ashawkey/torch-ngp},
    Title = {Torch-ngp: a PyTorch implementation of instant-ngp}
}
```

## Acknowledgements
This work is based on SuLvXiangXin's repo https://github.com/SuLvXiangXin/zipnerf-pytorch.git

- Thanks to [multinerf](https://github.com/google-research/multinerf) for amazing multinerf(MipNeRF360,RefNeRF,RawNeRF) implementation
- Thanks to [accelerate](https://github.com/huggingface/accelerate) for distributed training
- Thanks to [torch-ngp](https://github.com/ashawkey/torch-ngp) for super useful hashencoder
- Thanks to [Yurui Chen](https://github.com/519401113) for discussing the details of the paper.
