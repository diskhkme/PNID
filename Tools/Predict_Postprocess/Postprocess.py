import os.path
import pickle
import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from NMS import nms_process


json_result_filepath = "D:/Lib/Pytorch/SwinTransformer/work-dir/with_text/test_epoch_24.bbox.json"
test_img_dict_pkl_path = "D:/Data/PNID/EWP_Data/Drawing_Segment/scale_1_800_300_with_text/test_img_dict.pickle"
source_img_folder = "D:/Data/PNID/EWP_Data/Drawing/"

segment_params = [800, 800, 300, 300]

with open(json_result_filepath, "rb") as f:
    predict_result = json.load(f)

with open(test_img_dict_pkl_path, "rb") as f:
    test_img_dict = pickle.load(f)

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
for key, vals in converted_result.items():
    source_img_path = os.path.join(source_img_folder, f"{key}.jpg")
    img = cv2.imread(source_img_path)

    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for box in vals: # [[x, y, width, height], category, score]
        rect = patches.Rectangle((box[0][0], box[0][1]),
                                 box[0][2], box[0][3],
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()

for key, vals in converted_result.items():
    boxes, labels, scores = nms_process(vals, 0.5)

    print(boxes)