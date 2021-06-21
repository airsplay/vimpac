# Usage:
# To extract VQ-VAE tokens for UCF101 val split videos at 2 FPS, center crop 256x256, run:
#     $ export VIDEO_ROOT=/path/to/video/root;
#     $ export TOKEN_ROOT=/path/to/tokens;
#     $ export CUDA_VISIBLE_DEVICES=0;  # this split only works for a single GPU.
#     $ bash video2token/scripts/extract_tokens.sh ucf101 val 256 256 center 0 2
# You can append --debug or --data_ratio 0.01 to fast debugging
dset_name=$1  # kinetics400, ucf101, hmdb51, diving48, sthv2
split_name=$2  # train or val for [kinetics400, sthv2], train_val for [ucf101, hmdb51, diving48]
frame_shorter_size=$3  # first resize video shorter side to this, then crop. One if [128, 256]
frame_out_size=$4  # output frame size, set to one of [128, 256]
crop_type=$5  # [top, center, bottom]
use_hflip=$6  # [0, 1] whether to do horizontal flip
model_type=dalle
fps=$7  # frame per second

video_root_dir=${VIDEO_ROOT}/${dset_name}/videos
if [[ $dset_name == "ucf101" || $dset_name == "hmdb51" ]]; then
  split_name=train_val
  for i in 1 2 3; do
    for s in train val; do
      data_path+=(${VIDEO_ROOT}/${dset_name}/${dset_name}_${s}_split_${i}_videos.txt)
    done
  done
elif [[ $dset_name == "sthv2" || $dset_name == "diving48" || $dset_name == "kinetics400" ]]; then
  data_path=(${VIDEO_ROOT}/${dset_name}/${dset_name}_${split_name}_list_videos.txt)
else
  echo "error dset_name ${dset_name}"
  exit 1
fi

# token save dir
lmdb_save_dir=${TOKEN_ROOT}/${model_type}_${dset_name}_${split_name}_fps${fps}_hflip${use_hflip}
lmdb_save_dir=${lmdb_save_dir}_${frame_shorter_size}${crop_type}${frame_out_size} # filename contd

# others
batch_size=32

extra_args=()
if [[ ${use_hflip} == 1 ]]; then
  extra_args+=(--hflip)
fi

# start extraction
PYTHONPATH=$PYTHONPATH:. \
python video2token/extract_video_tokens.py \
--model_type ${model_type} \
--dset_name ${dset_name} \
--video_root_dir ${video_root_dir} \
--data_path ${data_path[@]} \
--fps ${fps} \
--frame_out_size ${frame_out_size} \
--frame_shorter_size ${frame_shorter_size} \
--crop_type ${crop_type} \
--lmdb_save_dir ${lmdb_save_dir} \
--batch_size ${batch_size} \
--num_workers 10 \
--fp16 \
${extra_args[@]} \
${@:8}
