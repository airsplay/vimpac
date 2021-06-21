# The name of this experiment.
DATASET=howto100m

name=small

# Save logs and models under snap; make backup.
output=snap/pretrain/$name
mkdir -p $output/src
cp -r vimpac $output/src/
cp -r *.py $output/src/
cp $0 $output/run.bash

CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:. \
    python vimpac/pretrain.py \
    --mlm-weight 1. \
    --nce-weight 1. \
    --nce-proj-layers 2 \
    --nce-proj-hid-dim 2048 \
    --model "efflayoutT,H|W" \
    --dataset-name $DATASET \
    --pos-emb hw_separation \
    --layers 6 \
    --hid-dim 512 \
    --frame-size 128 \
    --clip-len 5 \
    --num-test-clips 1 \
    --vocab-size 8192 \
    --lr 1e-3 \
    --beta1 0.9 \
    --beta2 0.98 \
    --clip-grad-norm 1. \
    --epochs 10 \
    --eval-per-epochs 1 \
    --save-per-epochs 9595 \
    --lr-scheduler linear \
    --warm-up 0.1 \
    --bs-per-gpu 8  \
    --batch-size 1024 \
    --mask-blocks 5 \
    --fp16 \
    --num-workers 2 \
    --output $output ${@:2} | tee -a $output/log.log
