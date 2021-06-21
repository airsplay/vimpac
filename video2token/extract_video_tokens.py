import random
import torch
import lmdb
import math
import time
import os
from os.path import join, exists
import numpy as np
from tqdm import tqdm
import pprint
from torch.utils.data.dataloader import DataLoader
from video2token.datasets.dataset_video_classification import VideoClassificationDataset
from video2token.utils.lmdb_utils import save_npz_to_lmdb
from video2token.utils.basic_utils import save_json
from video2token.load_dalle import load_clip_model
from dataclasses import dataclass


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--model_type", type=str, default="dalle", choices=["dalle"])
    parser.add_argument("--video_root_dir", type=str, help="path to dir storing video")
    parser.add_argument("--data_path", type=str, nargs="+", help="path or paths to annotation file")
    parser.add_argument("--dset_name", type=str, help="name of the dataset",
                        choices=VideoClassificationDataset.DATASET_NAMES)
    parser.add_argument("--lmdb_save_dir", type=str, help="path to dir storing lmdb files")
    parser.add_argument("--fps", type=int, default=2, help="fps for video")
    parser.add_argument("--frame_out_size", type=int, default=128, help="size of the output cropped video frames")
    parser.add_argument("--frame_shorter_size", type=int, default=160, help="shorter side size of the video, ")
    parser.add_argument("--hflip", action="store_true", help="use horizontal flip")
    parser.add_argument("--crop_type", default="center", choices=["none", "top", "center", "bottom"])
    parser.add_argument("--num_workers", type=int, default=10, help="#workers for data loading")
    parser.add_argument("--pin_memory", type=int, default=1, choices=[0, 1], help="pin cuda memory")
    parser.add_argument("--batch_size", type=int, default=64, help="#input images for the model")
    parser.add_argument("--mean", type=float, nargs=3, default=(0.5, 0.5, 0.5), help="image mean")
    parser.add_argument("--std", type=float, nargs=3, default=(0.5, 0.5, 0.5), help="image std")
    parser.add_argument("--np_dtype", type=str, default="int16", choices=["int16", "int32"],
                        help="which dtype to use for saving")

    # model
    parser.add_argument("--model_path", type=str, help="path storing a model checkpoint, "
                                                       "this does not take effect for --model_type dalle")
    # others
    parser.add_argument("--fp16", action="store_true", help="use mixed precision.")
    parser.add_argument("--no_cuda", action="store_true", help="do not use cuda")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--no_tqdm", action="store_true", help="do not use tqdm")
    # debug
    parser.add_argument("--data_ratio", type=float, default=1., help="percentage of data to use, 1: 100%, 0.1: 10%")
    parser.add_argument("--debug", action="store_true", help="break the loop")

    args = parser.parse_args()

    if args.debug or args.data_ratio != 1.:
        args.lmdb_save_dir = args.lmdb_save_dir + "_debug"

    if args.model_type == "dalle":
        args.model_path = None

    # check path & save args
    assert not exists(args.lmdb_save_dir) or len(os.listdir(args.lmdb_save_dir)) == 0, \
        f"dir {args.lmdb_save_dir} exists and not empty"
    os.makedirs(args.lmdb_save_dir, exist_ok=True)

    args.use_cuda = not args.no_cuda
    assert args.use_cuda, "the func `extract_single_video` is cuda only"

    save_json(vars(args), join(args.lmdb_save_dir, "args.json"), save_pretty=True)

    pprint.pprint(vars(args))

    if args.np_dtype == "int16":
        args.np_dtype = np.int16
    elif args.np_dtype == "int32":
        args.np_dtype = np.int32
    return args


def get_dataloader(args):
    dataset = VideoClassificationDataset(
        video_root_dir=args.video_root_dir,
        anno_path=args.data_path,
        fps=args.fps,
        size=args.frame_shorter_size,
        output_size=args.frame_out_size,
        hflip=args.hflip,
        crop_type=args.crop_type,
        data_ratio=args.data_ratio,
        mean=args.mean,
        std=args.std
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,  # a single video
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    return dataloader


def get_model(args):
    @dataclass
    class ARGS:
        model_path: str
        frame_size: int = 224

    model_args = ARGS(args.model_path, args.frame_out_size)
    model = load_clip_model(model_args).eval()
    if args.use_cuda:
        model.cuda()
    return model


@torch.no_grad()
def extract_single_video(model, video_frames, args):
    batch_size = args.batch_size
    num_batches = int(math.ceil(len(video_frames) / float(batch_size)))
    tokens = []
    for i in range(num_batches):
        _video_frames = video_frames[i * batch_size:(i + 1) * batch_size]
        if args.use_cuda:
            _video_frames = _video_frames.cuda()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            _tokens = model(_video_frames)
        tokens.append(_tokens.cpu())
    return torch.cat(tokens, dim=0).numpy()


def start_extract():
    args = get_args()
    set_random_seed(args.seed)

    dataloader = get_dataloader(args)
    model = get_model(args)

    env = lmdb.open(args.lmdb_save_dir, map_size=1024 ** 4)
    txn = env.begin(write=True)
    error_filepaths = []

    enumerator = enumerate(dataloader)
    if not args.no_tqdm:
        enumerator = tqdm(
            enumerator, total=len(dataloader), desc="extracting tokens")
    else:
        start_time = time.time()
    for idx, data in enumerator:
        video_frames, video_name, video_path, label_id = \
            data["video_frames"][0], data["video_name"][0], \
            data["video_path"][0], data["label_id"][0]
        # print(f"idx {idx}, video_name: {video_name}")
        # skip on loading error
        if video_frames.ndim < 4:  # normal data will have 4 dimensions
            error_filepaths.append(video_path)
            continue

        # forward
        try:
            video_tokens = extract_single_video(model, video_frames, args)  # np array
        except Exception as e:
            # skip on error
            print(f"Caught Exception {e}")
            continue

        save_npz_to_lmdb(
            lmdb_txn=txn, key=video_name,
            tokens=video_tokens.astype(args.np_dtype),
            label=np.array(label_id).astype(args.np_dtype)
        )

        # commit every 1000
        if idx % 100 == 0:
            txn.commit()
            txn = env.begin(write=True)

        if args.no_tqdm and idx % 100 == 0:
            elapsed_hours = (time.time() - start_time) / 3600.
            print(f"Finished {idx}/{len(dataloader)}, Time Elapsed: {elapsed_hours:.2f} hours")

        if args.debug and idx > 10:
            break

    txn.commit()
    env.close()

    with open(join(args.lmdb_save_dir, "err_filepaths.txt"), "w") as f:
        f.write("\n".join(error_filepaths))


if __name__ == '__main__':
    start_extract()
