import argparse
import logging
import random

import torch
import numpy as np
from eyewitness.mot.tracker import ObjectTracker
from eyewitness.mot.video import Mp4AsVideoData
from eyewitness.mot.evaluation import VideoTrackedObjects
from eyewitness.mot.visualize_mot import draw_tracking_result
from eyewitness.config import BoundedBoxObject

from tracker.multitracker import JDETracker
from utils.datasets import letterbox
from utils.log import logger
from utils.timer import Timer

parser = argparse.ArgumentParser(prog="naive_tracker.py")
parser.add_argument("--cfg", type=str, default="cfg/yolov3.cfg", help="cfg file path")
parser.add_argument(
    "--weights", type=str, default="weights/latest.pt", help="path to weights file"
)
parser.add_argument(
    "--img-size", type=int, default=(1088, 608), help="size of each image dimension"
)
parser.add_argument(
    "--iou-thres",
    type=float,
    default=0.5,
    help="iou threshold required to qualify as detected",
)
parser.add_argument(
    "--conf-thres", type=float, default=0.5, help="object confidence threshold"
)
parser.add_argument(
    "--nms-thres",
    type=float,
    default=0.4,
    help="iou threshold for non-maximum suppression",
)
parser.add_argument(
    "--min-box-area", type=float, default=200, help="filter out tiny boxes"
)
parser.add_argument("--track-buffer", type=int, default=30, help="tracking buffer")
parser.add_argument("--input-video", type=str, help="path to the input video")
parser.add_argument("--output-video", type=str, default="output.mp4")


class TowardRealtimeMOTracker(ObjectTracker):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.width, self.height = opt.img_size

    def track(self, video_data):
        """
        Parameters
        ----------
        video_data: VideoData
            the video data to be tracked
        Returns
        -------
        video_tracked_result: VideoTrackedObjects
            the tracked video result
        """
        timer = Timer()
        tracker = JDETracker(self.opt, frame_rate=video_data.frame_rate)

        # run tracking
        tracked_objects = VideoTrackedObjects()
        for idx in range(1, int(video_data.n_frames) + 1):
            if idx % 20 == 0:
                logger.info(
                    "Processing frame {} ({:.2f} fps)".format(
                        idx, 1.0 / max(1e-5, timer.average_time)
                    )
                )
            timer.tic()
            image_obj = video_data[idx]
            ori_w, ori_h = image_obj.size
            img, _, _, _ = letterbox(
                np.array(image_obj), height=self.height, width=self.width
            )

            # Normalize RGB
            img = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)
            img /= 255.0

            blob = torch.from_numpy(img).cuda().unsqueeze(0)
            online_targets = tracker.update(blob, (ori_h, ori_w, 3))

            for t in online_targets:
                x1, y1, w, h = t.tlwh
                x2, y2 = (x1 + w, y1 + h)
                object_id = t.track_id
                vertical = w / h > 1.6
                if w * h > opt.min_box_area and not vertical:
                    tracked_objects[idx].append(
                        BoundedBoxObject(
                            x1, y1, x2, y2, int(object_id), float(t.score), ""
                        )
                    )
            timer.toc()

        return tracked_objects


def get_spaced_colors(n):
    max_value = 16581375  # 255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]


if __name__ == "__main__":
    logger.setLevel(logging.INFO)

    opt = parser.parse_args()

    tracker = TowardRealtimeMOTracker(opt)
    mp4_as_video = Mp4AsVideoData(opt.input_video)

    color_list = get_spaced_colors(100)
    random.shuffle(color_list)
    result = tracker.track(mp4_as_video)

    draw_tracking_result(
        result, color_list, mp4_as_video, output_video_path=opt.output_video,
    )
