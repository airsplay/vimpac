import random

import torch


def mask_3d(codes: torch.Tensor,
            mask_blocks: int = 4,
            pad_token_id: int = 8193,
            mask_token_id: int = 8196,
            vocab_size: int = 8192,
            ):
    """

    :param codes:
    :param mask_blocks: If >0, it's the number of blocks.
                        If <0, it's uniform masking and the masking ratio is -{mask-blocks} / 1000.
    :param pad_token_id: Token id for [PAD]
    :param mask_token_id: Token id for [MASK]
    :param vocab_size:  Not used here.
    :return:
    """
    b, l, h, w = codes.shape

    masked_codes = codes.detach().clone()
    labels = codes.detach().clone()

    if mask_blocks < 0:
        # Apply random mask
        probability_matrix = torch.full(labels.shape, -mask_blocks / 1000, device=labels.device)
        mask = torch.bernoulli(probability_matrix).bool()
    else:
        mask = torch.zeros_like(codes, dtype=torch.bool)
        # Build the block mask
        for i in range(b):
            for j in range(mask_blocks):        # Number of masked blocks
                # l_block_len = random.randint(1, (l + 1) // 2)
                # h_block_len = random.randint(1, h // 2)
                # w_block_len = random.randint(1, w // 2)
                # l_block_len = random.randint(1, (l + 1) // 2)
                l_block_len = random.randint(1, l * 2 // 3)     # 5 --> 3, 10 --> 6, 15 --> 10
                h_block_len = random.randint(1, h // 2)         # 128 --> 64, 256 --> 128
                w_block_len = random.randint(1, w // 2)         # 128 --> 64, 256 --> 128

                l1 = random.randint(0, l - l_block_len)
                h1 = random.randint(0, h - h_block_len)
                w1 = random.randint(0, w - w_block_len)

                l2 = l1 + l_block_len
                h2 = h1 + h_block_len
                w2 = w1 + w_block_len

                mask[i, l1:l2, h1:h2, w1:w2] = 1

    # 1. the token is masked
    # 2. the original code is not padded
    mask = mask & (codes != pad_token_id)

    # Do not predict unmasked tokens
    labels[~mask] = -100

    # Set the code to mask id
    masked_codes[mask] = mask_token_id

    return masked_codes, labels


def mask_3d_rand_replace(codes: torch.Tensor,
                         mask_blocks: int = 4,
                         pad_token_id: int = 8193,
                         mask_token_id: int = 8196,
                         vocab_size: int = 8192,
                         ):
    b, l, h, w = codes.shape

    masked_codes = codes.detach().clone()
    labels = codes.detach().clone()

    if mask_blocks == -1:
        # Apply random mask
        assert False
        probability_matrix = torch.full(labels.shape, 0.15, device=labels.device)
        mask = torch.bernoulli(probability_matrix).bool()
    else:
        mask = torch.zeros_like(codes, dtype=torch.bool)
        # Build the block mask
        for i in range(b):
            for j in range(mask_blocks):        # Number of masked blocks
                l_block_len = random.randint(1, l * 2 // 3)     # 5 --> 3, 10 --> 6, 15 --> 10
                h_block_len = random.randint(1, h // 2)         # 128 --> 64, 256 --> 128
                w_block_len = random.randint(1, w // 2)         # 128 --> 64, 256 --> 128

                l1 = random.randint(0, l - l_block_len)
                h1 = random.randint(0, h - h_block_len)
                w1 = random.randint(0, w - w_block_len)

                l2 = l1 + l_block_len
                h2 = h1 + h_block_len
                w2 = w1 + w_block_len

                mask[i, l1:l2, h1:h2, w1:w2] = 1

                prob = random.random()
                if prob < 0.8:
                    masked_codes[i, l1:l2, h1:h2, w1:w2].fill_(mask_token_id)
                elif prob < 0.9:
                    masked_codes[i, l1:l2, h1:h2, w1:w2].random_(0, vocab_size)
                else:
                    # Do nothing to keep the original code
                    pass

    # A token need to be predicted if:
    # 1. the token is masked
    # 2. the original code is not padded
    need_predict = mask & (codes != pad_token_id)

    # Do not predict other tokens
    labels[~need_predict] = -100

    # Reset the pad tokens in the masked tokens
    masked_codes[codes == pad_token_id] = pad_token_id

    return masked_codes, labels
