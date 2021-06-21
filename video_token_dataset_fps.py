"""
Load extracted video tokens and their corresponding labels from LMDB file
"""
import lmdb
import os
import torch
import random
from torch.utils.data.dataset import Dataset
from torch import distributed
from tqdm import tqdm
from video2token.utils.lmdb_utils import load_npz_from_lmdb
from video2token.utils.basic_utils import get_basename_no_ext, flat_list_of_lists
from video_utils import pad_sequences_1d


def get_start_end_idx(video_size, clip_size, clip_idx, num_clips, start_idx_bound=None):
    """
    Args:
        video_size: int, #frames in video
        clip_size: int, #frames in clip
        clip_idx: int, if clip_idx = -1, random sampling. If
            clip_idx is larger than -1, uniformly split the video to `num_clips`
            clips, and select the start and end index of the `clip_idx`-th video
            clip.
        num_clips: int,  overall number of clips to uniformly sample from the
            given video for testing. Only used when clip_idx > 0.
        start_idx_bound: (int, int), always samples between the bound
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index.
    References:
        https://github.com/facebookresearch/SlowFast/blob/master/slowfast/datasets/decoder.py#L31
    """
    delta = max(video_size - clip_size, 0)

    if clip_idx == -1:
        # Actually, this formula support a naive bound (e.g., (0, 2147483647)).
        # We still use the None semantic for safety.
        if start_idx_bound is not None:
            l, r = start_idx_bound

            #                                 clip_size
            #  Video:            |------------|-------|
            #                    0          delta   video_size
            # r(delta)=    000000rrrrrrrrrrrrrddddddddddddddd
            # l(r, delta)= 000000llllllrrrrrrrrrrrrrrrrrrrrrr
            r = max(min(r, delta), 0)
            l = min(max(0, l), r)
        else:
            l, r = 0, delta

        # Random temporal sampling.
        start_idx = random.randint(l, r)

    else:
        # Uniformly sample the clip with the given index.
        # clip_idx ranges from [0, num_clips - 1]. Thus we divide it by (num_clilps - 1)
        start_idx = delta * clip_idx / max((num_clips - 1), 1)

    start_idx = int(start_idx)
    end_idx = start_idx + clip_size - 1

    return start_idx, end_idx


class VideoTokenDataset(Dataset):
    """
    If `is_train`:
        temporally - randomly sample a single clip of `num_frames` frames from the video;
        spatially - randomly sample a spatial crop, by randomly sampling a LMDB loaded from one of `lmdb_paths`.
    If not `is_train`:
        temporally - uniformly sample `num_test_clips` clips, each clip contains `num_frames` from the video;
        spatially - fixed to sample 3 spatial crops, [top, center, bottom], by using the corresponding 3 LMDBs

    lmdb_paths: list(str), each str is a path to an lmdb directory.
        The lmdbs are supposed to have the same set of keys.
    keys_to_use: list(str), which keys to use in the lmdb, if None use all.
    has_label: whether the lmdb file has data label stored
    num_frames: int, #frames to use for each video
    num_test_clips: int, overall number of clips to uniformly sample from the
        given video for testing.
    is_train: bool, whether in training mode. This affects the data loading strategy.
    padding: bool, if True, pad the sampled clip to `num_frames` with pad_token_id
    pad_token_id: int,
    """
    def __init__(self, lmdb_paths, keys_to_use=None, has_label=True, num_frames=10, sampled_frame_rate=1,
                 padding=False, pad_token_id=0, num_train_clips=1, relevance_range=None,
                 num_test_clips=5, is_train=True):
        # Data and Split
        self.lmdb_paths = lmdb_paths
        self.has_label = has_label
        self.keys = None
        self.keys_to_use = keys_to_use
        self.load_keys(lmdb_paths[0], keys_to_use=keys_to_use)

        # Clip configuration
        self.num_frames = num_frames            # Number of frames per clip
        self.padding = padding                  # Whether padding the clip with <pad_token_id>
        self.pad_token_id = pad_token_id
        self.is_train = is_train
        self.num_train_clips = num_train_clips
        self.num_test_clips = num_test_clips    # test only
        self.sampled_frame_rate = sampled_frame_rate        # How many frames are sampled from the whole clip.

        # Pre-training Specific (two-clip sampling)
        self.relevance_range = relevance_range

    def load_keys(self, lmdb_path, keys_to_use=None):
        # Preload the keys
        envs, _, lmdb_names = self._open_lmdb_files([lmdb_path])
        self.keys = self._get_keys(envs[lmdb_names[0]], keys_to_use=keys_to_use)
        envs[lmdb_names[0]].close()

    def set_pad_token_id(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __len__(self):
        """this is the length of a single lmdb file"""
        return len(self.keys)

    def __getitem__(self, index):
        # Actually open the env/txn at the first iterator.
        if not hasattr(self, "envs"):
            self.envs, self.txns, self.lmdb_names = self._open_lmdb_files(self.lmdb_paths)
        if self.is_train:
            return self._getitem_train(index)
        else:
            return (index, *self._getitem_test(index))

    def _getitem_train(self, index):
        """randomly sample a clip from a randomly sampled lmdb"""
        lmdb_key = self.keys[index]

        npz = load_npz_from_lmdb(self.sample_lmdb_txn()[0], lmdb_key)
        video_tokens = npz["tokens"]  # (#frames, 8, 8)
        video_tokens_list = self.temporal_random_sample(video_tokens, self.num_train_clips)  # (<=self.num_frames, 8, 8)
        video_tokens_list = [torch.from_numpy(video_tokens) for video_tokens in video_tokens_list]

        if self.padding:
            # concat_and_or_pad will reduce x with len(x) == 0 to x[0]
            video_tokens = self.concat_and_or_pad(video_tokens_list)  # (self.num_frames, 8, 8)
        else:
            assert self.num_train_clips == 1
            video_tokens = video_tokens_list[0]

        label_id = None
        if self.has_label:
            label_id = int(npz["label"])

        return video_tokens, label_id

    def _getitem_test(self, index):
        """uniformly sample `self.num_test_clips` clips each lmdb database"""
        lmdb_key = self.keys[index]
        npz_list = [load_npz_from_lmdb(self.txns[name], lmdb_key) for name in self.lmdb_names]
        # list of tensor (#frames, 8, 8), length = #video
        video_tokens_list = [torch.from_numpy(npz["tokens"]) for npz in npz_list]
        # list of tensor (<=self.num_frames, 8, 8), length = #video * #clips
        video_tokens = flat_list_of_lists([self.sample_frames(t) for t in video_tokens_list])
        if self.padding:
            video_tokens = self.concat_and_or_pad(video_tokens)
        label_id = None
        if self.has_label:
            label_id = int(npz_list[0]["label"])
        return video_tokens, label_id

    def concat_and_or_pad(self, clip_tokens_list):
        """Concat and or Pad a list of torch tensors
        Args:
            clip_tokens_list: list(tensor)
        Returns:
            array of shape (#frames, *) or (#clips, #frames, *)
        """
        if len(clip_tokens_list) == 1:  # at training or single clip test
            res_tokens = clip_tokens_list[0]
            if len(res_tokens) < self.num_frames:
                res_tokens, _ = pad_sequences_1d(
                    [res_tokens], pad_token_id=self.pad_token_id, fixed_length=self.num_frames)
                res_tokens = res_tokens.squeeze(0)
            return res_tokens  # (#frames, 8, 8)
        else:  # multi-clip testing
            res_tokens, _ = pad_sequences_1d(
                clip_tokens_list, pad_token_id=self.pad_token_id, fixed_length=self.num_frames)
            return res_tokens  # (#clips, #frames, 8, 8)

    def sample_lmdb_txn(self):
        lmdb_name = random.choice(self.lmdb_names)
        return self.txns[lmdb_name], lmdb_name

    def sample_frames(self, video_tokens):
        """Temporal sampling"""
        if self.is_train:  # random sample a clip from the video
            clip_tokens = self.temporal_random_sample(video_tokens)
            return clip_tokens  # (#frames, 8, 8)
        else:  # uniformly `self.num_test_clips` from the video
            clip_tokens_list = self.temporal_uniform_sample(video_tokens)
            return clip_tokens_list  # list(array of (#frames, 8, 8))

    def temporal_uniform_sample(self, video_tokens):
        """uniformly sample `self.num_test_clips` clips from the video.
        Args:
            video_tokens: np array, (#frames, *)
        """
        clip_tokens_list = []
        for clip_idx in range(self.num_test_clips):
            start_idx, end_idx = get_start_end_idx(
                len(video_tokens) + self.sampled_frame_rate - 1,    # This is the hack to include the last frames
                self.num_frames * self.sampled_frame_rate,          # The whole duration of the clip over the lmdb fps
                clip_idx=clip_idx,
                num_clips=self.num_test_clips
            )
            clip_tokens_list.append(video_tokens[start_idx: end_idx+1: self.sampled_frame_rate])
        return clip_tokens_list

    def temporal_random_sample(self, video_tokens, num_clips=1):
        """randomly sample a clip of self.num_frames frames from the video.
        Args:
            video_tokens: np array, (#frames, *)
            num_clips: int, number of training clips
        Return:
            [(#clip_len, *), (#clip_len, *), ...]  of length num_clips

        """
        video_tokens_list = []
        start_idx_bound = None

        for i in range(num_clips):
            start_idx, end_idx = get_start_end_idx(
                len(video_tokens) + self.sampled_frame_rate - 1,    # This is the hack to include the last frames
                self.num_frames * self.sampled_frame_rate,          # The whole duration of the clip over the lmdb fps
                clip_idx=-1,
                num_clips=None,
                start_idx_bound=start_idx_bound
            )
            video_tokens_list.append(video_tokens[start_idx: end_idx+1: self.sampled_frame_rate])

            if self.relevance_range is not None:
                # We assume that a relevance clip would be within the relevance range (e.g., 10 * clip_len).
                # The bound is over the start index.
                # If the range is None, we keep the bound always be None (i.e., no bound).
                start_idx_bound = (start_idx - self.relevance_range, start_idx + self.relevance_range)

        return video_tokens_list

    @staticmethod
    def _open_lmdb_files(lmdb_paths):
        """Open a list of lmdb files"""
        envs = {
            get_basename_no_ext(p): lmdb.open(p, readonly=True, create=False, lock=False)
            for p in lmdb_paths
        }
        # return `buffer` objects instead of bytestrings, this significantly improves performance
        txns = {name: env.begin(buffers=True) for name, env in envs.items()}
        names = list(txns.keys())
        return envs, txns, names

    def close_lmdb_files(self):
        for attr in ["envs", "txns", "lmdb_names"]:
            self.__dict__.pop(attr, None)

    @classmethod
    def key_preprocess(cls, key):
        return key

    def _get_keys(self, env, keys_to_use=None):
        """extract the subset of keys to use"""
        if distributed.get_rank() == 0:
            print("Gather available keys from LMDB file.")
        with env.begin(buffers=False) as tmp_txn:
            keys = sorted([k.decode() for k in list(tmp_txn.cursor().iternext(values=False))])  # list(str)
            # keys = sorted([k.decode() for k in tqdm(tmp_txn.cursor().iternext(values=False),
            #                                         total=env.stat()["entries"])])  # list(str)
        if keys_to_use is not None:
            keys_to_use = set(keys_to_use)  # check 'val in set' is O(1), 'val in list' is O(N)
            raw_len = len(keys)
            keys = [k for k in keys if self.key_preprocess(k) in keys_to_use]
            if distributed.get_rank() == 0:
                print(f"Selected {len(keys)} to use from {raw_len} keys, "
                      f"with keys_to_use {len(keys_to_use)}")
        return keys


class VideoTokenRandAugDataset(VideoTokenDataset):
    """ Dataset instance accepting shuffled and random augmented video tokens.
    Use `shuffle=False` when constructing the dataloader as the data has already been shuffled.
    num_epochs: int, #epochs this lmdb contains.
    """
    def __init__(self, lmdb_path, keys_to_use=None, num_epochs=100, num_frames=10,
                 padding=False, pad_token_id=0, num_test_clips=5):
        super(VideoTokenRandAugDataset, self).__init__(
            lmdb_paths=[lmdb_path], keys_to_use=keys_to_use, has_label=True, num_frames=num_frames,
            padding=padding, pad_token_id=pad_token_id, num_test_clips=num_test_clips,
            is_train=True
        )
        self.total_num_epochs = num_epochs  # not used

    def sample_lmdb_txn(self):
        """override `VideoTokenDataset.sample_lmdb_txn`
        """
        lmdb_name = self.lmdb_names[0]
        return self.txns[lmdb_name], lmdb_name

    @classmethod
    def key_preprocess(cls, key):
        """overwrite"""
        return "_".join(key.split("_")[2:])


def load_split_video_names(split_filepath):
    with open(split_filepath, "r") as f:
        video_names = [os.path.splitext(l.split()[0])[0] for l in f.readlines()]
    return set(video_names)


def main_test():

    import pprint
    # kinetics700, ucf101, hmdb51, howto100m, ssv2
    dset_name = "ssv2"
    # train_test for [hmdb51, ucf101]; [train, val] for [kinetics700, kinetics400, ssv2], train_val for howto100m
    split_name = "val"
    is_local = False  # whether on UNC local server or Google Cloud Platform.
    is_train = True  # True for training and False for test.
    padding = True
    pad_token_id = 0
    num_test_clips = 5
    num_frames = 10  # fps=2, 10 frames corresponds to a 5-seconds clip
    hflip_types = [0, 1] if is_train else [0]  # use hflip or not,
    crop_types = ["top", "center", "bottom"]
    crop_size = 128   # one of [128, 256]
    assert crop_size in [128, 256]
    factor = int(crop_size / 128)
    before_crop_sizes = [128 * factor, 160 * factor] if is_train else [128 * factor]

    video_token_root = "data/video_code"
    anno_root = "data/video_anno"

    # get lmdb paths
    lmdb_dir_paths = [
        video_token_root + f"/dalle_{dset_name}_{split_name}_fps2_hflip{hflip}_{before_crop_size}{crop_type}{crop_size}"
        for hflip in hflip_types for crop_type in crop_types for before_crop_size in before_crop_sizes
    ]

    lmdb_dir_paths = [e for e in lmdb_dir_paths if os.path.exists(e)]
    print(f"#lmdb paths: {len(lmdb_dir_paths)}, they are:\n{pprint.pformat(lmdb_dir_paths)}")
    if dset_name in ["kinetics700", "kinetics400"]:
        # splits are stored in different lmdbs
        lmdb_keys_to_use = None
    elif dset_name == "ucf101":
        # all splits are stored in the same lmdb file, so need to used keys_to_use to
        # get the separate split data
        val_split1_filepath = f"{anno_root}/ucf101/ucf101_val_split_1_videos.txt"
        lmdb_keys_to_use = load_split_video_names(val_split1_filepath)
    elif dset_name == "hmdb51":
        # all splits are stored in the same lmdb file, so need to used keys_to_use to
        # get the separate split data
        train_split1_filepath = f"{anno_root}/hmdb51/hmdb51_val_split_1_videos.txt"
        lmdb_keys_to_use = load_split_video_names(train_split1_filepath)
    elif dset_name == "howto100m":
        train_split_filepath = f"{anno_root}/howto100m/howto100m_videos_train_w_fake_label.csv"
        all_split_filepath = f"{anno_root}/howto100m/howto100m_videos_w_fake_label.csv"
        val_split_filepath = f"{anno_root}/howto100m/howto100m_videos_val_w_fake_label.csv"
        lmdb_keys_to_use = load_split_video_names(val_split_filepath)
    elif dset_name == "ssv2":
        # splits are stored in different lmdbs, no need to use `lmdb_keys_to_use`
        train_split_filepath = f"{anno_root}/ssv2/sthv2_train_list_videos.txt"
        val_split_filepath = f"{anno_root}/ssv2/sthv2_val_list_videos.txt"
        lmdb_keys_to_use = None
    else:
        raise ValueError
    video_tokens_dataset = VideoTokenDataset(
        lmdb_dir_paths, keys_to_use=lmdb_keys_to_use, has_label=True, num_frames=num_frames,
        padding=padding, pad_token_id=pad_token_id, num_test_clips=num_test_clips, is_train=is_train
    )
    tokens, label = video_tokens_dataset[0]
    # labels = list(set([int(e[1]) for e in video_tokens_dataset]))  # very slow
    # print(f"min label {min(labels)}, max label {max(labels)}")
    print(f"video tokens shape {tokens.shape}, label {label}")
    import ipdb; ipdb.set_trace()


def main_test_rand_aug():
    dset_name = "hmdb51"  # ucf101 or hmdb51
    is_local = False  # whether on UNC local server or Google Cloud Platform.
    padding = True
    pad_token_id = 0
    num_test_clips = 5
    num_frames = 10  # fps=2, 10 frames corresponds to a 5-seconds clip
    video_token_root = "/net/bvisionserver4/playpen10/jielei/data/vqvae_tokens" \
        if is_local else "/mnt/data/data/video_code"
    lmdb_dir_path = video_token_root + f"/dalle_{dset_name}_train_test_randaug_e100_fps2_128-160_crop128"
    anno_root = "/net/bvisionserver4/playpen10/jielei/data/mmaction2_data" \
        if is_local else "/mnt/data/data/video_anno"
    if dset_name == "kinetics700":
        # splits are stored in different lmdbs
        lmdb_keys_to_use = None
    elif dset_name == "ucf101":
        # all splits are stored in the same lmdb file, so need to used keys_to_use to
        # get the separate split data
        val_split1_filepath = f"{anno_root}/ucf101/ucf101_train_split_1_videos.txt"
        lmdb_keys_to_use = load_split_video_names(val_split1_filepath)
    elif dset_name == "hmdb51":
        # all splits are stored in the same lmdb file, so need to used keys_to_use to
        # get the separate split data
        train_split1_filepath = f"{anno_root}/hmdb51/hmdb51_train_split_1_videos.txt"
        lmdb_keys_to_use = load_split_video_names(train_split1_filepath)
    else:
        raise ValueError
    video_tokens_rand_aug_dataset = VideoTokenRandAugDataset(
        lmdb_dir_path, keys_to_use=lmdb_keys_to_use, num_frames=num_frames, padding=padding,
        pad_token_id=pad_token_id, num_test_clips=num_test_clips
    )
    tokens, label = video_tokens_rand_aug_dataset[0]
    print(f"video tokens shape {tokens.shape}, label {label}")
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    from torch import distributed

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=1,
        rank=0
    )
    main_test()
    # main_test_rand_aug()
