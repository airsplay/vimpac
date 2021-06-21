# Video2Token

In this directory, we provide code and instructions on how to prepare datasets for training. In particular, we cover the following aspects: (1) prepare and preprocess videos and annotations with [MMAction2](https://github.com/open-mmlab/mmaction2); (2) discretize videos using [DALL-E VQ-VAE](https://github.com/openai/DALL-E). 

## Prepare and Preprocess Videos and Annotations
In this section, we will go through the process of downloading videos and annotations, and preprocessing them into required formats. This section will not use any code in this repo, we fully rely on MMAction2. 

### Requirements
Install [MMAction2](https://github.com/open-mmlab/mmaction2) following [the official instructions](https://mmaction2.readthedocs.io/en/latest/install.html). For your convenience, we summarize the procedure below. If you encounter any issue, please refer the official instructions for additional information.
```shell script
# 1, create conda environment
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
# 2, install PyTorch 
conda install pytorch torchvision -c pytorch
# 3, install MMAction2 and its dependencies via mim package manager
# this may take a while
pip install git+https://github.com/open-mmlab/mim.git
mim install mmaction2
# 4, clone MMAction2, and cd into the project directory. 
# This would be your default working directory.
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
git checkout 0a6fde1abb8403f1f68b568f5b4694c6f828e27e .
```  

### Getting Started
All the downstream datasets used in our project, including Sthv2 (SSV2), Kinetics-400, Diving48, HMDB51, UCF101 can be found in the [Supported Datasets](https://mmaction2.readthedocs.io/en/latest/supported_datasets.html) section. Please follow these instructions to prepare the videos and annotations. Here we use UCF101 as an example to illustrate the process. 

1. Download videos and annotations

    ```shell script
    cd tools/data/ucf101/
    bash download_annotations.sh
    bash download_videos.sh  
    ```

    You may need to install unrar first for decompressing the video file, `sudo apt-get install unrar`. The annotations and videos will be stored at `$MMAction2/data/ucf101` by default. If you desire to store the data to somewhere else, you may soft link a directory to be `$MMAction2/data`  with `ln -s /path/to/your/dir $MMAction2/data`. 

2. Generate video file list

    ```shell script
    bash generate_videos_filelist.sh
    ```
   
   This generates files that contain list of paths to the video files and their labels. Note that our video data loader directly loads raw videos, there is no need to extract RGB frames.

3. Check Directory Structure

    After the first two steps, you should expect to see the following folder structure:
    
    ```
    mmaction2
    ├── mmaction
    ├── tools
    ├── configs
    ├── data
    │   ├── ucf101
    │   │   ├── ucf101_{train,val}_split_{1,2,3}_rawframes.txt
    │   │   ├── ucf101_{train,val}_split_{1,2,3}_videos.txt
    │   │   ├── annotations
    │   │   ├── videos
    │   │   │   ├── ApplyEyeMakeup
    │   │   │   │   ├── v_ApplyEyeMakeup_g01_c01.avi
    │   │   │   ├── ...
    │   │   │   ├── YoYo
    │   │   │   │   ├── v_YoYo_g25_c05.avi
    ```

For pretraining videos, i.e., HowTo100m videos, please follow the [official website](https://www.di.ens.fr/willow/research/howto100m/) to download the videos. 

## Discrete Videos with DALL-E VQ-VAE
In this section, we detail the process of converting raw videos into discrete tokens via [DALL-E VQ-VAE](https://github.com/openai/DALL-E).

### Requirements
```shell script
# 1, create conda environment
conda create -n dalle python=3.7 -y
conda activate dalle
# 2, install PyTorch 
conda install pytorch torchvision -c pytorch
# 3, install DALL-E
pip install DALL-E
# 4, install ffmpeg and other dependencies
conda install -c conda-forge ffmpeg
pip install ffmpeg-python lmdb tqdm
```  

### Getting Started
We provide an easy-to-use script [extract_tokens.sh](./scripts/extract_tokens.sh) to extract VQ-VAE tokens from raw videos. The extracted tokens are stored in LMDB files. Below is a template to use this script.

```shell script
# at project root
export VIDEO_ROOT=/path/to/video/root # this will be $MMAction2/data
export TOKEN_ROOT=/path/to/tokens  # somewhere to save the VQ-VAE tokens
export CUDA_VISIBLE_DEVICES=0  # only a single GPU is supported.
bash video2token/scripts/extract_tokens.sh DATASET_NAME SPLIT_NAME FRAME_SHORTER_SIDE CROP_SIZE CROP_TYPE USE_HFLIP FPS
```

The input arguments are:

| Argument             | Definition                     | Common Values                                                                                      |
|----------------------|--------------------------------|-------------------------------------------------------------------------------------------------------|
| `DATASET_NAME`       | dataset name            | `kinetics400`, `ucf101`, `hmdb51`, `diving48`, `sthv2`                                                |
| `SPLIT_NAME`         | split name                     | `train` or `val` for [`kinetics400`, `sthv2`], `train_val` for [`ucf101`, `hmdb51`, `diving48`] |
| `FRAME_SHORTER_SIDE` | frame shorter side length      | `128`, `160`, `256`, `320`                                                                                          |
| `CROP_SIZE`          | output frame crop size         | `128` for `FRAME_SHORTER_SIDE` in [`128`, `160`], `256` for [`256`, `320`]                                                                                          |
| `CROP_TYPE`          | crop location                  | `top`, `center`, `bottom`                                                                             |
| `USE_HFLIP`          | use horizontal flip | `0`, `1`                                                                                              |
| `FPS`                | #frames per second             | integer, e.g., `2` or `4`                                                                                  |
 
To extract VQ-VAE tokens for UCF101 val split videos at 2 FPS, center crop 256x256 from frames with shorter side resized (keep aspect ratio) to 320, run:
```shell script
# at project root
export VIDEO_ROOT=/path/to/video/root # this will be $MMAction2/data
export TOKEN_ROOT=/path/to/tokens  # somewhere to save the VQ-VAE tokens
export CUDA_VISIBLE_DEVICES=0  # only single GP extraction is supported.
bash video2token/scripts/extract_tokens.sh ucf101 val 320 256 center 0 2
```
The extracted tokens will be stored in an LMDB file located at `$TOKEN_ROOT/dalle_ucf101_train_val_fps2_hflip0_320center256`, and they are ready to be used for training. 


## Acknowledgement
This code used resources from [MMAction2](https://github.com/open-mmlab/mmaction2), 
[DALL-E](https://github.com/openai/DALL-E), [video_feature_extractor](https://github.com/antoine77340/video_feature_extractor), [ffmpeg-python](https://github.com/kkroening/ffmpeg-python). The code is implemented using [PyTorch](https://github.com/pytorch/pytorch). We thank the authors for open-sourcing their awesome projects.



 
