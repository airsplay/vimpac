# VIMPAC: Video Pre-Training via Masked Token Prediction and Contrastive Learning

This is a release of our [VIMPAC paper](http://arxiv.org/abs/2106.11250) to illustrate the implementations.
The pretrained checkpoints and scripts will be soon open-sourced in HuggingFace transformers.

Authors: [Hao Tan](https://www.cs.unc.edu/~airsplay/), [Jie Lei](https://www.cs.unc.edu/~jielei/), [Thomas Wolf](https://thomwolf.io/), [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)

## Data Preprocessing
Please refer to [video2token](video2token) folder for the detailed README file.

For pre-training, the dataset is usually large, and we suggest to use `FPS=2` during extraction.
For downstream tasks, we suggest using `FPS=16` that enables a higher frame rate for short videos.

We recommend to store the data locally at  `data/video_tokens`.
If different paths are used, please specify the path of `VIDEO_CODE_PATHS` and `VIDEO_ANNO_PATHS` in [vimpac/data.py](vimpac/data.py).


## Pre-Trained Weights
We provide the pre-trained weights with their links.
Please download the pre-trained weight and extract them under `snap/`.

- Small model (GitHub Version), Frame 128; This model is exactly trained from this GitHub version, 
following the instructions and scripts in [pre-training](#pre-training).
    Link: [https://nlp.cs.unc.edu/data/vimpac_snap/small.zip](https://nlp.cs.unc.edu/data/vimpac_snap/small.zip)
- Large model, Frame 128
    Link: [https://nlp.cs.unc.edu/data/vimpac_snap/large_frame128.zip](https://nlp.cs.unc.edu/data/vimpac_snap/large_frame128.zip)
- Large model, Frame 256
    Link: [https://nlp.cs.unc.edu/data/vimpac_snap/large_frame256.zip](https://nlp.cs.unc.edu/data/vimpac_snap/large_frame256.zip)


## Pre-Training
The default pre-training uses the [HowTo100M](https://www.di.ens.fr/willow/research/howto100m/) dataset.
The pre-training data could be switched to [Kinetics-700](https://deepmind.com/research/open-source/kinetics) and other 
datasets by specifying the `--dataset-name` argument.
We have validated that the mask-then-predict task works reasonablely well on Kinetics-700 datasets.
However, the average length of video clips inside K-700 is 10 seconds thus not sure supporting the long-range contrastive learning.

### Small Model
We first provide the script to pre-train a small model (6 layers, 512 dimensions, 256 frame-size, and 5 clip length):
```shell
bash scripts/pretrain/small.sh 0,1,2,3
```

We here annotate some essential arguments inside the pre-training scripts.
For a full descriptions for all the arguments, please check [param.py](vimpac/param.py)
```
python vimpac/pretrain.py \
    --mlm-weight 1. \       # The weight for masked-language-modeling loss
    --nce-weight 1. \       # The weight of constrative learning loss
    --nce-proj-layers 2 \   # Number of layers in contrastive learning's projection head.
    --nce-proj-hid-dim 2048 \   # Hidden dimensions in contrastive learning's projection head.
    --model "efflayoutT,H|W" \  # The model layout, common options: `T,H|W', `T,H,W', `T|H|W'  
    --dataset-name $DATASET \   # Name of datasets, options: "kinetics400", "ucf101", "hmdb51", "ssv2", "howto100m", "diving48"
    --layers 6 \            # Number of layers in the backbone model
    --hid-dim 512 \         # Hidden dimensions of the backbone model
    --frame-size 128 \      # Spatial frame size of the input; 128 --> 16 codes; 256 --> 32 codes.
    --clip-len 5 \          # Temporal clip length for each input.
    --lr 1e-3 \             # Learning rate
    --beta1 0.9 \           # Adam's beta1
    --beta2 0.98 \          # Adam's beta2. This hyperparatmers needs to be changed for large model.
    --lr-scheduler linear \     # Learning rate decay methods, options: `linear', `cosine', 'constant'
    --warm-up 0.1 \         # Warmup steps/ratio. If <1, it's the ratio. Otherwise, it's the actual steps.
    --mask-blocks 5     # Number of masking blocks
```

We also provide two debugging options:
```shell
# bash scripts/pretrain/small.sh 0,1,2,3 --tqdm        # Show progress bar.
# bash scripts/pretrain/small.sh 0,1,2,3 --debug       # Only run a few steps per epoch.
```

### Large Model
We follow [BERT](https://export.arxiv.org/abs/1810.04805) to pre-train our large model in two stages. 
The first stage pretrains for 90 epochs using frame-size 128 and clip-length 5.
The second stage pretrains for 10 epochs using frame-size 256 and clip-length 5.

Scripts for the first stage:
```shell
bash scripts/pretrain/large.sh 0,1,2,3
```
Then we could directly run the script for the second stage without any further changes. 
It will load the last snapshot from the first stage, do interpolation for larger spatial size, and continue pre-training.
```shell
bash scripts/pretrain/large_frame256cont.sh 0,1,2,3
```

## Fine-Tuning
After run the pre-training in [pre-training](#pre-training) or download the pre-trained weights from [pre-trained-weights](#pre-trained-weights),
we fine-tune the models on several downstream tasks.
The arguments in these scripts are consistent with the hyperparameters in the paper.
Please refer to Table 11 and Table 12 of our paper for a detailed list of all these hyperparameters.  

### SSV2
```shell
bash scripts/finetune/small_ssv2.sh 0,1,2,3
```

### Diving48
```shell
bash scripts/finetune/small_diving48.sh 0,1,2,3
```

### UCF101
```shell
bash scripts/finetune/small_ucf101.sh 0,1,2,3
```

### HMDB51
```shell
bash scripts/finetune/small_hmdb51.sh 0,1,2,3
```

### Change the Input Shape
Following [ViT](https://github.com/google-research/vision_transformer),
we support the use of different input sizes from pre-training by interpolating the positional embedding.
This is done by passing the `--different-shape` option.
Otherwise, an error will pop up if the fine-tuning input shape is different from the pre-training.
A larger input shape generally improves the results. We here take SSV2 as an example.

Longer clip length (10; default 5):
```shell
bash scripts/finetune/small_ssv2.sh 0,1,2,3 --different-shape --clip-len 10 --bs-per-gpu 4
```
Long clip length (10; default 5) + higher frame rate (4; default 2)
```shell
bash scripts/finetune/small_ssv2.sh 0,1,2,3 --different-shape --clip-len 10 --frame-rate 4 --bs-per-gpu 4
```
Long clip length (10; default 5) + higher frame rate (4; default 2) + larger input size (256; default 128).
Please also make sure that VQ-VAE code with input-size 256 has been extracted as in [Pre-processing](video2token).
```shell
bash scripts/finetune/small_ssv2.sh 0,1,2,3 --different-shape --clip-len 10 --frame-rate 4 --frame-size 256 --bs-per-gpu 2
```

### Large Models
We provide scripts to run large models. Frame 128:
```shell
bash scripts/finetune/large_frame128_ucf101.sh 0,1,2,3
```
Frame 256:
```shell
bash scripts/finetune/large_frame256_ucf101.sh 0,1,2,3
```
The input shape could be changed as in [change input shape](#change-the-input-shape). Our final model use the scripts of:
```shell
bash scripts/finetune/large_frame256_ucf101.sh 0,1,2,3 --different-shape --clip-len 10 --frame-rate 4 --frame-size 256 --bs-per-gpu 2
```


## Acknowledgement
This  work  was  granted  access  to  the  HPC  resources  of  IDRIS  under  the  allocation  20XX-AD011011621R1 made by GENCI. 
We thank Teven Le Scao and Victor Sanh for their help on the way.



