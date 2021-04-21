import os.path
import pickle
import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from NMS import greedy_nms_process
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO


json_result_filepath = "D:/Libs/Pytorch/SwinTransformer/workdir/wo_text/test_epoch_24.bbox.json"
test_img_dict_pkl_path = "D:/Test_Models/PNID/EWP_Data/Drawing_Segment/scale_1_800_300_wo_text/test_img_dict.pkl"
ground_truth_pkl_path = "D:/Test_Models/PNID/EWP_Data/Drawing_Segment/scale_1_800_300_wo_text/ground_truth.pkl"
symbol_dict_pkl_path = "D:/Test_Models/PNID/EWP_Data/Drawing_Segment/scale_1_800_300_wo_text/symbol_dict.pkl"
source_img_folder = "D:/Test_Models/PNID/EWP_Data/Drawing/"
greedy_nms_iou_threshold = 0.5

segment_params = [800, 800, 300, 300]

with open(json_result_filepath, "rb") as f:
    predict_result = json.load(f)

with open(test_img_dict_pkl_path, "rb") as f:
    test_img_dict = pickle.load(f)

with open(ground_truth_pkl_path, "rb") as f:
    ground_truth = pickle.load(f)

with open(symbol_dict_pkl_path, "rb") as f:
    symbol_dict = pickle.load(f)

converted_result = {}
for result in predict_result:
    seg_img_file = [key for key,val in test_img_dict.items() if val == result["image_id"]][0] # dict를 쓰긴 하지만 1:1임
    source_img_filename = seg_img_file[0:seg_img_file.find("_")]
    source_img_row = int(seg_img_file[seg_img_file.find("_")+1:seg_img_file.rfind("_")])
    source_img_column = int(seg_img_file[seg_img_file.rfind("_")+1:seg_img_file.rfind(".")])
    box_coord = result['bbox']
    category = result['category_id']
    score = result['score']

    converted_bbox = [box_coord[0] + segment_params[2] * source_img_row,
                      box_coord[1] + segment_params[3] * source_img_column,
                      box_coord[2], box_coord[3]]
    if source_img_filename in converted_result:
        converted_result[source_img_filename].append([converted_bbox, category, score])
    else:
        converted_result[source_img_filename] = []
        converted_result[source_img_filename].append([converted_bbox, category, score])



# Visualize Data
# for key, vals in converted_result.items():
#     source_img_path = os.path.join(source_img_folder, f"{key}.jpg")
#     img = cv2.imread(source_img_path)
#
#     fig, ax = plt.subplots(1)
#     ax.imshow(img)
#     for box in vals: # [[x, y, width, height], category, score]
#         rect = patches.Rectangle((box[0][0], box[0][1]),
#                                  box[0][2], box[0][3],
#                                  linewidth=1, edgecolor='r', facecolor='none')
#         ax.add_patch(rect)
#     plt.show()

for key, vals in converted_result.items():
    img_name = key
    #boxes, labels, scores = greedy_nms_process(vals, greedy_nms_iou_threshold)

    source_img_path = os.path.join(source_img_folder, f"{key}.jpg")
    img = cv2.imread(source_img_path)

    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for box in boxes: # [xmin, ymin, xmax, ymax]
        rect = patches.Rectangle((int(box[0]), int(box[1])),
                                 int(box[2]-box[0]), int(box[3]-box[1]),
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()


    # gt_data = ground_truth[f"{img_name}.jpg"]
    # gt_boxes = np.array([x[1:] for x in gt_data])
    # gt_labels = np.array([symbol_dict[x[0]] for x in gt_data])
    #
    # total_prediction = boxes.shape[0]
    # total_gt = gt_boxes.shape[0]