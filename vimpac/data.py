import collections

import torch
import torch.distributed
from torch.utils.data import Dataset, DataLoader

from video_token_dataset_fps import load_split_video_names
from vimpac.modeling_utils import PAD_TOKEN_ID

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


VIDEO_CODE_PATHS = [
    "data/video_tokens",
    "/path/to/tokens",
]

VIDEO_ANNO_PATHS = [
    "data/video_anno",
    "/path/to/video/root",
]


def find_path(path_list):
    import os
    for path in path_list:
        if os.path.exists(path):
            return path
    else:
        print(f"NO PATH FIND for {path_list}")


class Evaluator:
    """
    An independent logic for the evaluation. It ensures the correctness of the code.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.id2label = None

    def build_id2label(self):
        self.id2label = {}
        for k in range(len(self.dataset)):
            i, _, label = self.dataset[k]
            self.id2label[i] = label
        del self.dataset

    def evaluate(self, id2pred, force=True):
        # Lazy building
        if self.id2label is None:
            self.build_id2label()

        # Actual evaluation
        if force:
            assert len(id2pred) == len(self.id2label)
        correct = 0
        amount = len(id2pred)
        for i, pred in id2pred.items():
            label = self.id2label[i]
            if label == pred:
                correct += 1
        return correct / max(amount, 1)


def get_dataset(
        dset_name: str = "ucf101",
        split: str = "train",
        split_id: int = 1,
        frame_size: int = 128,
        clip_len: int = 10,
        frame_rate: int = None,
        num_train_clips: int = 1,
        num_test_clips: int = 5,
        bs: int = 32,
        shuffle: bool = False,
        drop_last: bool = False,
        dist: bool = False,
        num_workers: int = 4,
        is_train: bool = None,
        train_aug: bool = None,
        only_one_aug: bool = False,
        relevance_range: int = None,
):
    """
    :param frame_rate: The sampled frame rate.
    :param is_train: if is_train, return [b, l, h, w]
                     if False, return [b, num_test_clilps, l, h, w]
    :param train_aug: if true, use all 12 augmentations;
                      if False, use 3 canonical augmentations. (spatial crops)
                      if None, will be the same as "is_train"
    :param rand_aug: if True, use random aug for 100 epochs.
    :return:
    """

    if is_train is None:
        is_train = (split == "train")

    video_token_root = find_path(VIDEO_CODE_PATHS)
    anno_root = find_path(VIDEO_ANNO_PATHS)

    if split_id != 1:
        print(f"Use split with id {split_id}")

    if dset_name == "kinetics400":
        # splits are stored in different lmdbs
        lmdb_keys_to_use = None

        if split == "train":
            split_name = "train"
        elif split == "valid":
            split_name = "val"
        else:
            assert False

        num_classes = 400
    elif dset_name == "ucf101":
        # all splits are stored in the same lmdb file, so need to used keys_to_use to
        # get the separate split data
        if split == "train":
            split_filepath = f"{anno_root}/ucf101/ucf101_train_split_{split_id}_videos.txt"
        elif split == "valid":
            split_filepath = f"{anno_root}/ucf101/ucf101_val_split_{split_id}_videos.txt"
        else:
            raise ValueError(f"Split {split} is not valid.")
        lmdb_keys_to_use = load_split_video_names(split_filepath)

        split_name = "train_test"
        num_classes = 101
    elif dset_name == "hmdb51":
        # all splits are stored in the same lmdb file, so need to used keys_to_use to
        # get the separate split data
        if split == "train":
            split_filepath = f"{anno_root}/hmdb51/hmdb51_train_split_{split_id}_videos.txt"
        elif split == "valid":
            split_filepath = f"{anno_root}/hmdb51/hmdb51_val_split_{split_id}_videos.txt"
        else:
            raise ValueError(f"Split {split} is not valid.")
        lmdb_keys_to_use = load_split_video_names(split_filepath)

        split_name = "train_test"
        num_classes = 51
    elif dset_name == "ssv2":
        lmdb_keys_to_use = None
        if split == "train":
            split_name = "train"
        elif split == "valid":
            split_name = "val"
        else:
            assert False
        num_classes = 174
    elif dset_name == "diving48":
        lmdb_keys_to_use = None
        if split == "train":
            split_name = "train"
        elif split == "valid":
            split_name = "val"
        else:
            assert False
        num_classes = 48
    elif dset_name == "howto100m":
        lmdb_keys_to_use = None
        split_name = "train_val"
        num_classes = -1  # HT 100M does not have classes.
    else:
        raise ValueError(f"No such dataset {dset_name}")

    assert frame_size == 128 or frame_size == 256

    # Augmentations
    train_aug = is_train if train_aug is None else train_aug        # If not specifying train_aug, follow `is_train`

    if only_one_aug:
        hflip_types = [0]
        crop_types = ["center"]
        before_crop_sizes = [frame_size]
    elif dset_name == "ssv2":
        hflip_types = [0]               # No horizon flip for SSV2
        crop_types = ["top", "center", "bottom"] if train_aug else ["center"]
        before_crop_sizes = [frame_size, frame_size * 160 // 128] if train_aug else [frame_size]
    else:
        hflip_types = [0, 1] if train_aug else [0]  # use hflip or not,
        crop_types = ["top", "center", "bottom"]
        before_crop_sizes = [frame_size, frame_size * 160 // 128] if train_aug else [frame_size]

    # If we want to sample at a different frame rate, rather than frame rate 2.
    # We will load the lmdb with high frame rate (e.g., 16) and sparse sample frames from it.
    # This option is only provided for the downstream datasets.
    if frame_rate is not None:
        from video_token_dataset_fps import VideoTokenDataset
        lmdb_fps = 16
        # Simulate the frame rate by, for every K frames, sample a frame in clip
        sampled_frame_rate = max(int(lmdb_fps / frame_rate), 1)
        kwargs = {"sampled_frame_rate": sampled_frame_rate}
    else:
        # from video_token_dataset import VideoTokenDataset
        # kwargs = {}
        # lmdb_fps = 2
        from video_token_dataset_fps import VideoTokenDataset
        lmdb_fps = 2
        # Simulate the frame rate by, for every K frames, sample a frame in clip
        sampled_frame_rate = 1
        kwargs = {"sampled_frame_rate": sampled_frame_rate}

    # get lmdb paths
    lmdb_dir_paths = [
        video_token_root + f"/dalle_{dset_name}_{split_name}_fps{lmdb_fps}_hflip{hflip}_{before_crop_size}{crop_type}{frame_size}"
        for hflip in hflip_types for crop_type in crop_types for before_crop_size in before_crop_sizes
    ]

    video_tokens_dataset = VideoTokenDataset(
        lmdb_dir_paths, keys_to_use=lmdb_keys_to_use, has_label=True, num_frames=clip_len,
        padding=True, pad_token_id=PAD_TOKEN_ID,
        num_train_clips=num_train_clips, num_test_clips=num_test_clips,
        relevance_range=relevance_range,
        is_train=is_train,
        **kwargs
    )

    if dist:
        sampler = torch.utils.data.distributed.DistributedSampler(video_tokens_dataset, shuffle=shuffle)
        shuffle = None  # sampler option is mutually exclusive with shuffle
    else:
        sampler = None  # no sampler for non-distributed training.

    data_loader = DataLoader(
        video_tokens_dataset, batch_size=bs,
        shuffle=shuffle, sampler=sampler,  # Used to distributed
        drop_last=drop_last, pin_memory=True,
        num_workers=num_workers,
    )

    return DataTuple(dataset=video_tokens_dataset, loader=data_loader, evaluator=Evaluator(video_tokens_dataset)), num_classes
