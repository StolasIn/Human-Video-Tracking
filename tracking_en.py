import argparse
import os
import os.path as osp
import time
import cv2
import torch
import numpy as np

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess
from yolox.utils.visualize import plot_tracking
from tracker_class import MyTracker
from yolox.tracking_utils.timer import Timer
from wbf import *

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./videos/palace.mp4", help="path to images or video"
    )
    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c1", "--ckpt1", default=None, type=str, help="ckpt for eval")
    parser.add_argument("-c2", "--ckpt2", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )

    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=60, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


class Predictor(object):
    def __init__(
        self,
        model1,
        model2,
        exp,
        device = torch.device("cpu"),
    ):
        self.model1 = model1
        self.model2 = model2
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            timer.tic()
            outputs1 = self.model1(img)
            outputs2 = self.model2(img)
            outputs1 = postprocess(
                outputs1, self.num_classes, self.confthre, self.nmsthre
            )
            outputs2 = postprocess(
                outputs2, self.num_classes, self.confthre, self.nmsthre
            )
            outputs = self.fuse_result(outputs1, outputs2)
            
        return outputs, img_info

    def fuse_result(self, outputs1, outputs2):
        outputs1 = outputs1[0].cpu().numpy().astype(np.float64)
        outputs2 = outputs2[0].cpu().numpy().astype(np.float64)
        
        boxes_list = [outputs1[:, :4], outputs2[:, :4]]
        scores_list = [outputs1[:, 4] * outputs1[:, 5], outputs2[:, 4] * outputs2[:, 5]]
        labels_list = [outputs1[:, 6], outputs2[:, 6]]

        # 結合兩個模型的結果 (weighted)
        weights = [1, 2]
        boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=self.nmsthre, skip_box_thr=self.confthre, conf_type='box_and_model_avg')
        scores = np.expand_dims(scores, 1)
        dummy_score = np.ones_like(scores)
        labels = np.expand_dims(labels, 1)
        
        outputs = np.concatenate((boxes, scores, dummy_score, labels), 1)
        outputs = torch.from_numpy(outputs).unsqueeze(0)
        return outputs

def imageflow_demo(predictor, current_time, args):
    cap = cv2.VideoCapture(args.path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    os.makedirs('video_output', exist_ok=True)
    save_path = osp.join('./video_output', args.path.split("/")[-1])

    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = MyTracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    s = set()

    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val == False:
            break

        outputs, img_info = predictor.inference(frame, timer)

        # 如果有得到結果，就顯示 bbox
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp1.test_size, img_info['raw_img'])
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                s.add(tid)
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
            )

        # 反之就顯示原始影像
        else:
            timer.toc()
            online_im = img_info['raw_img']

        vid_writer.write(online_im)
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
        
        frame_id += 1
    logger.info(f"#human in the video : {len(s)}")


def main(exp1, exp2, args):
    if not args.experiment_name:
        args.experiment_name = exp1.exp_name

    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp1.test_conf = args.conf
    if args.nms is not None:
        exp1.nmsthre = args.nms

    model1 = exp1.get_model().to(args.device)
    model2 = exp2.get_model().to(args.device)
    model1.eval()
    model2.eval()

    ckpt_file1 = args.ckpt1
    ckpt_file2 = args.ckpt2
    logger.info("loading checkpoint")
    ckpt1 = torch.load(ckpt_file1, map_location=args.device)
    ckpt2 = torch.load(ckpt_file2, map_location=args.device)
    model1.load_state_dict(ckpt1["model"])
    model2.load_state_dict(ckpt2["model"])
    logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model1 = fuse_model(model1)
        model2 = fuse_model(model2)

    predictor = Predictor(model1, model2, exp1, args.device)
    current_time = time.localtime()
    imageflow_demo(predictor, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp1 = get_exp(args.exp_file, args.name)
    exp2 = get_exp(args.exp_file, args.name)

    main(exp1, exp2, args)
