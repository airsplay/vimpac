import os
from pathlib import Path
import warnings

from torch import nn
import torch.nn.functional as F

import sys
import time

import torch
from torchvision.utils import make_grid, save_image


# DALLE requirment
from dall_e import map_pixels, unmap_pixels
from dall_e import load_model as load_model_from_dalle


class DALLEExtractor(nn.Module):
    def __init__(self, device, check_steps=5):
        super(DALLEExtractor, self).__init__()

        MODEL_SAVE_DIR = Path("./dalle_ckpts/")    # you can download the weights here

        if (MODEL_SAVE_DIR / "encoder.pkl").exists():
            self.enc = load_model_from_dalle(str(MODEL_SAVE_DIR / "encoder.pkl"), device)
        else:
            print("Downloading DALLE encoder from https://cdn.openai.com/dall-e/encoder.pkl")
            self.enc = load_model_from_dalle("https://cdn.openai.com/dall-e/encoder.pkl", device)

        if check_steps > 0:
            if (MODEL_SAVE_DIR / "decoder.pkl").exists():
                self.dec = load_model_from_dalle(str(MODEL_SAVE_DIR / "decoder.pkl"), device)
            else:
                print("Downloading DALLE decoder from https://cdn.openai.com/dall-e/decoder.pkl")
                self.dec = load_model_from_dalle("https://cdn.openai.com/dall-e/decoder.pkl", device)

        self.image_size = 256
        self.counter = 0
        self.check_steps = check_steps

    def forward(self, img):
        b, c, h, w = img.shape
        self.counter += 1

        # Interpolate it to correct size of the model if it mismatches
        # if h != self.image_size or w != self.image_size:
        #     warnings.simplefilter('always', UserWarning)
        #     warnings.warn(f"The input height {h} width {w} is not same to the {self.image_size}"
        #                   f" keep the input size of {h, w}")s

        # Apply DALLE normalization (from [-1, 1] --> [0, 0.9];
        #   first map [-1, 1] --> [0, 1], and then use DALLE's pixel mappings from [0, 1] to [0, 0.9]
        img = map_pixels(img * 0.5 + 0.5)

        # Run the model
        # with torch.no_grad():
        z_logits = self.enc(img)                # B, channel, H, W   --> B, num_codes, H, W
        z = torch.argmax(z_logits, dim=1)      # B, num_codes, H, W --> B, H, W

        # For the first few runs, we will misc the reconstruction quality.
        check_rec = False
        if check_rec and self.counter < self.check_steps:
            z_onehot = F.one_hot(z, num_classes=self.enc.vocab_size).permute(0, 3, 1, 2).float()

            x_stats = self.dec(z_onehot).float()
            img_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))

            argmax_mse_loss = F.mse_loss(img, img_rec)
            if argmax_mse_loss.item() > 0.08:
                warnings.warn(f"The reconstruction loss {argmax_mse_loss.item()} the larger than 0.08."
                              f" Please be careful with it.")

            time_str = time.strftime("%Y_%m_%d_%H_%M_%S")
            os.makedirs("snap/debug/debug_video_tokens/", exist_ok=True)
            debug_img_path = f"snap/debug/debug_video_tokens/" \
                             f"{time_str}_top_raw_vs_bottom_recon_{self.counter}.jpg"
            n_img_per_row = 3
            img_rec = img_rec[:n_img_per_row]  # (#frames, 3, height, width)
            img = img[:n_img_per_row]  # (#frames, 3, height, width)
            img = unmap_pixels(img)
            grid = make_grid(torch.cat([img, img_rec], dim=0), nrow=n_img_per_row)
            save_image(grid, debug_img_path)
            print(f"Saved reconstructed images at {debug_img_path}")

        return z


def load_clip_model(args, device="cuda", **kwargs):
    """

    :param args: the args containing "model_path" and "frame_size"
        **The model will change the frame_size to the correct input_size**
    :param device: could either be str / torch.Device
    :return: a extractor (nn.Module) which take batch input.
        Usage: extractor(imgs)  # imgs: torch[batch_size, channel, height, width]
                                # return: torch[batch_size, height, width]

    Example:
        extractor = load_model(
            args,
            device="cpu",
        )

        # Fake Image Test:
        fake_images = torch.zeros((2, 3, args.frame_size, args.frame_size))
        fake_images.uniform_(-1, 1)
        code = extractor(fake_images)
    """
    args.frame_size = 256
    extractor = DALLEExtractor(device, **kwargs)

    return extractor


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class ARGS():
        model_path: str
        frame_size: int

    args = ARGS(
        "debug",
        224)
    print(args)

    device = torch.device("cuda")
    extractor = load_clip_model(args, device=device)

    # Fake Image Test:
    for _ in range(10):
        fake_images = torch.zeros((2, 3, args.frame_size, args.frame_size)).to(device)
        fake_images.uniform_(-1, 1)
        code = extractor(fake_images)
        print(code.shape)
