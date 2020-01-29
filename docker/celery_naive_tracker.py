import os
import argparse
import random

from celery import Celery
from eyewitness.mot.video import Mp4AsVideoData
from eyewitness.mot.visualize_mot import draw_tracking_result
from naive_tracker import TowardRealtimeMOTracker, get_spaced_colors


BROKER_URL = os.environ.get('broker_url', 'amqp://guest:guest@rabbitmq:5672')
celery = Celery('tasks', broker=BROKER_URL)

args = argparse.Namespace()
args.cfg = os.environ.get('cfg', 'cfg/yolov3.cfg')
args.weights = os.environ.get('weights', 'weights/latest.pt')
img_size = os.environ.get('img_size', (1088, 608))
if isinstance(img_size, str):
    img_size = (int(i) for i in img_size.split(','))
args.img_size = img_size
args.iou_thres = float(os.environ.get('iou_thres', 0.5))
args.conf_thres = float(os.environ.get('conf_thres', 0.5))
args.nms_thres = float(os.environ.get('nms_thres', 0.4))
args.min_box_area = float(os.environ.get('min_box_area', 200))
args.track_buffer = float(os.environ.get('track_buffer', 600))

TRACKER = TowardRealtimeMOTracker(args)


@celery.task(name='track_video')
def track_video(params):
    input_video = params.get('input_video', 'input.mp4')
    output_video = params.get('output_video', 'output.mp4')
    mp4_as_video = Mp4AsVideoData(input_video)

    color_list = get_spaced_colors(100)
    random.shuffle(color_list)
    result = TRACKER.track(mp4_as_video)

    draw_tracking_result(
        result, color_list, mp4_as_video, output_video_path=output_video,
    )
