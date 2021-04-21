import os
import random
import pickle
from Data_Generator.image_segmentation import segment_images_in_dataset
from Common.symbol_io import read_symbol_txt
from Data_Generator.write_coco_annotation import write_coco_annotation

# 학습 데이터 생성 코드. 도면을 train/test/val로 나누고, 각 set의 이미지를 분할하여 sub_img들로 만들어 저장함
# 이때, train의 경우 심볼(또는 옵션에 따라 심볼+텍스트)가 존재하지 않는 도면은 저장하지 않음
# 단 test/val 도면의 경우 심볼이 존재하지 않아도 저장함
# TODO: Scaling 기능 추가

base_dir = "D:/Test_Models/PNID/EWP_Data/"
drawing_dir = base_dir + "Drawing"
drawing_segment_dir = base_dir + "Drawing_Segment/dataset_0"
symbol_xml_dir = base_dir + "SymbolXML"
text_xml_dir = base_dir + "TextXML"
test_drawings = ["KNU-A-22300-001-04", "KNU-A-36420-014-03",
                "KNU-A-71710-003-02", "KNU-B-11600-001-03",
                "KNU-B-11600-002-03", "KNU-B-36420-019-04",
                "KNU-B-36610-002-05", "KNU-B-36610-004-01",
                "KNU-B-36610-005-03"]
symbol_txt_path = base_dir + "EWP_SymbolClass_sym_only.txt"

include_text_as_class = False
train_ratio = 0.9
segment_params = [800, 800, 300, 300] # width_size, height_size, width_stride, height_stride

symbol_dict = read_symbol_txt(symbol_txt_path)
if include_text_as_class == True:
    symbol_dict["text"] = len(symbol_dict.items()) + 1

xml_paths_without_test = [os.path.join(symbol_xml_dir, x) for x in os.listdir(symbol_xml_dir) if x.split(".")[0] not in test_drawings]
test_xmls = [os.path.join(symbol_xml_dir, f"{x}.xml") for x in test_drawings]

random.shuffle(xml_paths_without_test)
train_count = int(len(xml_paths_without_test)*train_ratio)
train_xmls = xml_paths_without_test[0:train_count]
val_xmls = xml_paths_without_test[train_count:]

val_annotation_data = segment_images_in_dataset(val_xmls, drawing_dir, drawing_segment_dir, segment_params, text_xml_dir, symbol_dict, include_text_as_class, "val")
write_coco_annotation(os.path.join(drawing_segment_dir,"val.json"), val_annotation_data, symbol_dict, segment_params)

test_annotation_data = segment_images_in_dataset(test_xmls, drawing_dir, drawing_segment_dir, segment_params, text_xml_dir, symbol_dict, include_text_as_class, "test")
write_coco_annotation(os.path.join(drawing_segment_dir,"test.json"), test_annotation_data, symbol_dict, segment_params)

train_annotation_data = segment_images_in_dataset(train_xmls, drawing_dir, drawing_segment_dir, segment_params, text_xml_dir, symbol_dict,  include_text_as_class, "train")
write_coco_annotation(os.path.join(drawing_segment_dir,"train.json"), train_annotation_data, symbol_dict, segment_params)

