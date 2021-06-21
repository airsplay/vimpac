import random
import torch
import numpy as np


def pad_sequences_1d(sequences, pad_token_id=0, dtype=torch.long, device=torch.device("cpu"), fixed_length=None):
    """ Pad a single-nested list or a sequence of n-d array (torch.tensor or np.ndarray)
    into a (n+1)-d array, only allow the first dim has variable lengths.
    Args:
        sequences: list(n-d tensor or list)
        pad_token_id: int, used as padding
        dtype: np.dtype or torch.dtype
        device:
        fixed_length: pad all seq in sequences to fixed length. All seq should have a length <= fixed_length.
            return will be of shape [len(sequences), fixed_length, ...]
    Returns:
        padded_seqs: ((n+1)-d tensor) padded with zeros
        mask: (2d tensor) of the same shape as the first two dims of padded_seqs,
              1 indicate valid, 0 otherwise
    Examples:
        >>> test_data_list = [[1,2,3], [1,2], [3,4,7,9]]
        >>> pad_sequences_1d(test_data_list, dtype=torch.long)
        >>> test_data_3d = [torch.randn(2,3,4), torch.randn(4,3,4), torch.randn(1,3,4)]
        >>> pad_sequences_1d(test_data_3d, dtype=torch.float)
        >>> test_data_list = [[1,2,3], [1,2], [3,4,7,9]]
        >>> pad_sequences_1d(test_data_list, dtype=np.float32)
        >>> test_data_3d = [np.random.randn(2,3,4), np.random.randn(4,3,4), np.random.randn(1,3,4)]
        >>> pad_sequences_1d(test_data_3d, dtype=np.float32)

    References:
        https://github.com/jayleicn/TVRetrieval/blob/master/utils/tensor_utils.py
    """
    if isinstance(sequences[0], list):
        if "torch" in str(dtype):
            sequences = [torch.tensor(s, dtype=dtype, device=device) for s in sequences]
        else:
            sequences = [np.asarray(s, dtype=dtype) for s in sequences]

    extra_dims = sequences[0].shape[1:]  # the extra dims should be the same for all elements
    lengths = [len(seq) for seq in sequences]
    if fixed_length is not None:
        max_length = fixed_length
    else:
        max_length = max(lengths)
    if isinstance(sequences[0], torch.Tensor):
        assert "torch" in str(dtype), "dtype and input type does not match"
        padded_seqs = torch.zeros((len(sequences), max_length) + extra_dims, dtype=dtype, device=device)
        padded_seqs.fill_(pad_token_id)
        mask = torch.zeros((len(sequences), max_length), dtype=torch.float32, device=device)
    else:  # np
        assert "numpy" in str(dtype), "dtype and input type does not match"
        padded_seqs = np.zeros((len(sequences), max_length) + extra_dims, dtype=dtype)
        padded_seqs = padded_seqs.fill(pad_token_id)
        mask = np.zeros((len(sequences), max_length), dtype=np.float32)

    for idx, seq in enumerate(sequences):
        end = lengths[idx]
        padded_seqs[idx, :end] = seq
        mask[idx, :end] = 1
    return padded_seqs, mask  # , lengths


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

        # linear sampling
        # if start_idx_bound is not None:
        #     l0, r0 = start_idx_bound
        #     mid = (l0 + r0) // 2
        #     def ratio(x):
        #         if x == mid:                            # make sure the new x is not the same as x
        #             return 0
        #         elif x < mid:
        #             return (x - l0 + 0.5) / (mid - l0)        # l0 - 0.5 --> 0, mid --> 1
        #         elif x > mid:
        #             return (x - r0 - 0.5) / (mid - r0)        # r0 + 0.5 --> 0, mid --> 1
        #     while random.random() > ratio(start_idx):
        #         start_idx = random.randint(l, r)
    else:
        # Uniformly sample the clip with the given index.
        start_idx = delta * clip_idx / max(num_clips, 1)

    start_idx = int(start_idx)
    end_idx = start_idx + clip_size - 1

    return start_idx, end_idx

