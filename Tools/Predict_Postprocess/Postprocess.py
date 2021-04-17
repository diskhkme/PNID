import pickle
import json

json_result_filepath = "D:/Libs/Pytorch/SwinTransformer/Swin-Transformer-Object-Detection/workdir/test_epoch_12/.bbox.json"
test_img_dict_pkl_path = "D:/Test_Models/PNID/EWP_Data/Drawing_Segment/scale0.5_800_300_wo_text/test_img_dict.pickle"
segment_params = [800, 800, 300, 300]

with open(json_result_filepath, "rb") as f:
    predict_result = json.load(f)

with open(test_img_dict_pkl_path, "rb") as f:
    test_img_dict = pickle.load(f)

converted_result = {}
for result in predict_result:
    seg_img_file = [key for key,val in test_img_dict.items() if val == result["image_id"]][0] # dict를 쓰긴 하지만 1:1임
    source_img_file = seg_img_file[0:seg_img_file.find("_")]
    source_img_row = int(seg_img_file[seg_img_file.find("_")+1:seg_img_file.rfind("_")])
    source_img_column = int(seg_img_file[seg_img_file.rfind("_")+1:seg_img_file.rfind(".")])
    box_coord = result['bbox']
    category = result['category_id']
    score = result['score']

    converted_bbox = [box_coord[0] + segment_params[2] * source_img_row,
                      box_coord[1] + segment_params[3] * source_img_column,
                      box_coord[2], box_coord[3]]
    if seg_img_file in converted_result:
        converted_result[seg_img_file].append([converted_bbox, category, score])
    else:
        converted_result[seg_img_file] = []
        converted_result[seg_img_file].append([converted_bbox, category, score])