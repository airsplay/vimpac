# The name of this experiment.
DATASET=ssv2
name=small

# Save logs and models under snap; make backup.
output=snap/${DATASET}/${name}
mkdir -p $output/src
cp -r vimpac $output/src/
cp -r *.py $output/src
cp $0 $output/run.bash

CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:. \
    python vimpac/finetune.py \
    --clip-len 5 \
    --frame-rate 2 \
    --bs-per-gpu 4   \
    --frame-size 128 \
    --lr 1e-4 \
    --different-shape \
    --num-test-clips 1 \
    --epochs 22 \
    --eval-per-epochs 8 \
    --model "efflayoutT,H|W" \
    --pos-emb hw_separation \
    --dataset-name $DATASET \
    --load snap/pretrain/$name/last/classifier.pt \
    --layers 6 \
    --hid-dim 512 \
    --last-test-clips 2,5,10 \
    --vocab-size 8192 \
    --clip-grad-norm 1. \
    --lr-scheduler linear \
    --last-dropout 0.0 \
    --weight-decay 0.01 \
    --beta1 0.9 \
    --beta2 0.999 \
    --warm-up 0.1 \
    --batch-size 128 \
    --fp16 \
    --num-workers 2 \
    --output $output ${@:2} | tee $output/log.log

