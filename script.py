from __future__ import print_function
import os
import sys
import cv2
import time
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from itertools import product as product
import numpy as np
from math import ceil
from models.config import *
from models.retinaface import RetinaFace

class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
    
class Count(object):

    def __init__(self, images_path ="images", cpu = True, evaluation = False, save_image = False,
                 top_k = 5000, keep_top_k = 750, vis_thres = 0.6, nms_threshold = 0.4, 
                 confidence_threshold = 0.02):
        super(Count, self).__init__()
        self.images_path = images_path
        self.cpu = cpu
        self.evaluation = evaluation
        self.save_image = save_image
        self.top_k = top_k
        self.keep_top_k = keep_top_k
        self.vis_thres = vis_thres
        self.nms_threshold = nms_threshold
        self.confidence_threshold = confidence_threshold

    def py_cpu_nms(self, dets, thresh):
        """Pure Python NMS baseline."""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def decode(self, loc, priors, variances):
        boxes = torch.cat((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def decode_landm(self, pre, priors, variances):
        landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                            ), dim=1)
        return landms

    def check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True

    def remove_prefix(self, state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def load_model(self, model, pretrained_path, load_to_cpu):
        if load_to_cpu:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        self.check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model

    def count_faces(self):
        print("------------------")
        if len(sys.argv) >= 2:
            self.images_path = sys.argv[1]
        total_time = 0
        total_precision = 0
        torch.set_grad_enabled(False)

        #choose network
        cfg = cfg_mnet
        # cfg = cfg_re50

        # net and model
        net = RetinaFace(cfg=cfg, phase = 'test')
        net = self.load_model(net, "./weights/mobilenet0.25_Final.pth", self.cpu)
        net.eval()
        cudnn.benchmark = True
        device = torch.device("cpu" if self.cpu else "cuda")
        net = net.to(device)
        resize = 1

        # obtain image path
        img_list = []
        if os.path.isdir(self.images_path):
            img_list = os.listdir(self.images_path)
        else:
            img_list.append(self.images_path)

        # testing begin
        all_faces = 0
        all_counts = 0
        for p in img_list:
            count = 0
            if os.path.isdir(self.images_path):
                img_raw = cv2.imread(os.path.join(self.images_path, p), cv2.IMREAD_COLOR)
            if not os.path.isdir(self.images_path):
                img_raw = cv2.imread(p, cv2.IMREAD_COLOR)
            img = np.float32(img_raw)
            im_height, im_width, _ = img.shape
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(device)
            scale = scale.to(device)
            begin = time.time()
            loc, conf, landms = net(img)  # forward pass
            end = time.time()
            eve_time = (end - begin) * 1000
        
            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(device)
            prior_data = priors.data
            boxes = self.decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = self.decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                    img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                    img.shape[3], img.shape[2]])
            scale1 = scale1.to(device)
            landms = landms * scale1 / resize
            landms = landms.cpu().numpy()

            # ignore low scores
            inds = np.where(scores > self.confidence_threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:self.top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = self.py_cpu_nms(dets, self.nms_threshold)
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets = dets[keep, :]
            landms = landms[keep]

            # keep top-K faster NMS
            dets = dets[:self.keep_top_k, :]
            landms = landms[:self.keep_top_k, :]
            dets = np.concatenate((dets, landms), axis=1)
            for b in dets:
                    if b[4] > self.vis_thres:
                        count += 1
            # show image
            if self.save_image:
                for b in dets:
                    if b[4] < self.vis_thres:
                        continue
                    text = "{:.4f}".format(b[4])
                    b = list(map(int, b))
                    cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                    cx = b[0]
                    cy = b[1] + 12
                    cv2.putText(img_raw, text, (cx, cy),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                    # landms
                    cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                    cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                    cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                    cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                    cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
                    cv2.imwrite(f"demo.jpg", img_raw)
            print(p + " " + str(count))

            #eval
            true_face = p.split("_")[0]
            eve_precision = int(true_face) / count
            all_faces += int(true_face)
            all_counts += count
            total_time += eve_time
            total_precision += eve_precision
        
        ave_time = total_time / len(img_list)
        ave_precision = total_precision / len(img_list)
        for item in sys.argv:
            if "evaluation" == item:
                self.evaluation = True
        if self.evaluation:
            print("Precision: {:.2f}%".format(ave_precision*100))
            print("Average Time: {:.3f} ms".format(ave_time))
        
        return all_faces == all_counts

#program entry
if __name__ == '__main__':
    obj = Count()
    obj.count_faces()
    
