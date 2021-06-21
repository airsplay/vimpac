from os.path import join, splitext
from tqdm import tqdm
from video2token.datasets.dataset_base import VideoDatasetBase


class VideoClassificationDataset(VideoDatasetBase):
    DATASET_NAMES = ["hmdb51", "ucf101", "kinetics700", "howto100m", "sthv2", "kinetics400", "diving48"]

    def __init__(self, video_root_dir, anno_path,
                 fps: int = 2, size: int = 224, output_size: int = 224,
                 crop_type: str = "center", hflip: bool = False, data_ratio: float = 1.,
                 mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.video_root_dir = video_root_dir
        datalist = self.load_anno(anno_path)
        super(VideoClassificationDataset, self).__init__(
            datalist, fps=fps, size=size, output_size=output_size,
            crop_type=crop_type, hflip=hflip,
            data_ratio=data_ratio, mean=mean, std=std
        )

    def __getitem__(self, index):
        data = self.datalist[index]
        video_frames = self.load_video_from_path(data["video_path"])
        return dict(
            video_frames=video_frames,
            video_name=data["video_name"],
            video_path=data["video_path"],
            label_id=data["label_id"]
        )

    def load_anno(self, anno_paths):
        """
        Args:
            anno_paths: list(str), each path leads to a .txt file where each line is
                "drawing/HaGIDVRRKhM_0-10000.mp4 167"  # path to video and its label index
                "flic_flac/Outdoor-Turnen_flic_flac_f_cm_np1_ri_med_2.avi 14"
        """
        lines = []
        for anno_path in anno_paths:
            with open(anno_path, "r") as f:
                lines += f.readlines()
        datalist = []
        collected_video_names = set()
        for line in tqdm(lines, desc="Converting annotation"):
            video_base_path, cls_id = line.split()
            video_name = splitext(video_base_path)[0]
            if video_name in collected_video_names:
                continue  # skip
            collected_video_names.add(video_name)
            datalist.append(dict(
                video_name=video_name,
                video_path=join(self.video_root_dir, video_base_path),
                label_id=int(cls_id)
            ))
        print(f"Collected {len(datalist)} unique videos.")
        return datalist
