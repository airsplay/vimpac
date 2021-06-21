# The name of this experiment.
DATASET=howto100m

name=large

# Save logs and models under snap; make backup.
output=snap/pretrain/$name
mkdir -p $output/src
cp -r vimpac $output/src/
cp -r *.py $output/src/
cp $0 $output/run.bash

CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:. \
    python video_pretrain_new/pretrain_nce.py \
    --mlm-weight 1. \
    --nce-weight 1. \
    --halve-mlp \
    --halve-att \
    --nce-proj-layers 3 \
    --nce-proj-hid-dim 4096 \
    --model "efflayoutT,H|W" \
    --dataset-name $DATASET \
    --pos-emb hw_separation \
    --layers 24 \
    --hid-dim 1024 \
    --frame-size 256 \
    --clip-len 5 \
    --num-test-clips 1 \
    --vocab-size 8192 \
    --lr 3e-4 \
    --beta1 0.9 \
    --beta2 0.98 \
    --clip-grad-norm 1. \
    --epochs 100 \
    --interval 10 \
    --eval-per-epochs 1 \
    --save-per-epochs 1 \
    --lr-scheduler linear \
    --warm-up 0.05 \
    --bs-per-gpu 1  \
    --batch-size 1024 \
    --mask-blocks 6 \
    --resume \
    --fp16 \
    --num-workers 2 \
    --output $output ${@:2} | tee -a $output/log.log

