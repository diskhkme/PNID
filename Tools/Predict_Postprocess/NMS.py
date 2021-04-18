import numpy as np
import torch
from mmcv.ops import nms, soft_nms

def nms_process(bboxes, iou_threshold):
    """
    :param bboxes: [[x,y,with,height],category,score]
    :param iou_threshold:
    :return:
    """
    bbox = torch.zeros((len(bboxes),4))
    score = torch.zeros((len(bboxes),))
    categories = torch.zeros((len(bboxes),))



    for i in range(len(bboxes)):
        bbox[i,:] = torch.Tensor([float(bboxes[i][0][0]),
                              float(bboxes[i][0][1]),
                              float(bboxes[i][0][0]+bboxes[i][0][2]),
                              float(bboxes[i][0][1]+bboxes[i][0][3])]) # [x1,y1,x2,y2]로 변환
        score[i] = float(bboxes[i][2])
        categories[i] = int(bboxes[i][1])

    dets, inds = nms(bbox,score,iou_threshold)
    inds = inds.numpy()
    return bbox[inds,:].numpy(), categories[inds].numpy(), score[inds].numpy()

