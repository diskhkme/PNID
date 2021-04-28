import os
import random
import pickle
from Data_Generator.generate_segmented_data import generate_segmented_data
from Common.symbol_io import read_symbol_txt
from Data_Generator.write_coco_annotation import write_coco_annotation

# 학습 데이터 생성 코드. 도면을 train/test/val로 나누고, 각 set의 이미지를 분할하여 sub_img들로 만들어 저장함
# 이때, train의 경우 심볼(또는 옵션에 따라 심볼+텍스트)가 존재하지 않는 도면은 저장하지 않음
# 단 test/val 도면의 경우 심볼이 존재하지 않아도 저장함

base_dir = "D:/Test_Models/PNID/EWP_Data/"
drawing_dir = base_dir + "Drawing"
drawing_segment_dir = base_dir + "Drawing_Segment/dataset_5_text_rot"
symbol_xml_dir = base_dir + "SymbolXML"
text_xml_dir = base_dir + "TextXML_All_Corrected"
train_drawings = ['KNU-A-22300-001-02', 'KNU-A-36120-001-01', 'KNU-A-36120-001-02',
                  'KNU-A-36120-001-03', 'KNU-A-36420-011-02', 'KNU-A-36420-012-01',
                  'KNU-A-36420-012-02', 'KNU-A-36420-012-03', 'KNU-A-36420-012-04',
                  'KNU-A-36420-012-05', 'KNU-A-36420-014-01', 'KNU-A-36420-014-02',
                  'KNU-A-36420-014-04', 'KNU-A-36420-018-01', 'KNU-A-36420-018-02',
                  'KNU-A-36420-018-03', 'KNU-A-36420-018-04', 'KNU-A-71120-001-01',
                  'KNU-A-71120-001-02', 'KNU-A-71120-001-03', 'KNU-A-71120-002-01',
                  'KNU-A-71120-002-02', 'KNU-A-71710-001-01', 'KNU-A-71710-001-02',
                  'KNU-A-71710-003-03', 'KNU-B-11600-001-01', 'KNU-B-11600-001-02',
                  'KNU-B-11600-001-04', 'KNU-B-11600-002-01', 'KNU-B-15100-002-01',
                  'KNU-B-15100-002-02', 'KNU-B-15100-002-03', 'KNU-B-36130-001-01',
                  'KNU-B-36130-001-02', 'KNU-B-36130-001-03', 'KNU-B-36130-002-01',
                  'KNU-B-36130-002-02', 'KNU-B-36130-003-02', 'KNU-B-36130-004-01',
                  'KNU-B-36130-004-02', 'KNU-B-36300-001-01', 'KNU-B-36300-001-02',
                  'KNU-B-36300-001-03', 'KNU-B-36420-015-01', 'KNU-B-36420-015-02',
                  'KNU-B-36420-015-03', 'KNU-B-36420-015-04', 'KNU-B-36420-015-05',
                  'KNU-B-36420-019-02', 'KNU-B-36420-019-03', 'KNU-B-36610-002-01',
                  'KNU-B-36610-002-02', 'KNU-B-36610-002-03', 'KNU-B-36610-002-04',
                  'KNU-B-36610-002-06', 'KNU-B-36610-004-02', 'KNU-B-36610-004-03',
                  'KNU-B-36610-004-04', 'KNU-B-36610-004-05', 'KNU-B-36610-005-01',
                  'KNU-B-36610-005-02', 'KNU-B-36610-005-04', 'KNU-B-36610-005-05',
                  'KNU-B-36610-005-06', 'KNU-B-36610-005-07']
val_drawings = ['KNU-A-22300-001-01', 'KNU-A-22300-001-03', 'KNU-A-36420-011-01',
                'KNU-A-71710-003-01', 'KNU-B-11600-002-02', 'KNU-B-11600-003-01',
                'KNU-B-36130-003-01', 'KNU-B-36420-019-01']
test_drawings = ["KNU-A-22300-001-04", "KNU-A-36420-014-03", "KNU-A-71710-003-02",
                 "KNU-B-11600-002-03", "KNU-B-36420-019-04", "KNU-B-36610-002-05",
                 "KNU-B-36610-004-01", "KNU-B-36610-005-03"]
ignore_drawing = ["KNU-B-11600-001-03"]
symbol_txt_path = base_dir + "EWP_SymbolClass_sym_only.txt"

include_text_as_class = True # Text를 별도의 클래스로 포함할 것인지 {"text"}
include_text_orientation_as_class = True # 세로 문자열을 또다른 별도의 클래스로 포함할 것인지 {"text_rotated"}

segment_params = [800, 800, 300, 300] # width_size, height_size, width_stride, height_stride
drawing_resize_scale = 0.5

symbol_dict = read_symbol_txt(symbol_txt_path, include_text_as_class, include_text_orientation_as_class)

xml_paths_without_test = [os.path.join(symbol_xml_dir, x) for x in os.listdir(symbol_xml_dir) if x.split(".")[0] not in test_drawings and x.split(".")[0] not in ignore_drawing]

train_xmls = [os.path.join(symbol_xml_dir, f"{x}.xml") for x in train_drawings]
val_xmls = [os.path.join(symbol_xml_dir, f"{x}.xml") for x in val_drawings]
test_xmls = [os.path.join(symbol_xml_dir, f"{x}.xml") for x in test_drawings]

# # Random Shuffle
# train_ratio = 0.9
#
# random.Random(1).shuffle(xml_paths_without_test)
# train_count = int(len(xml_paths_without_test)*train_ratio)
# train_xmls = xml_paths_without_test[0:train_count]
# val_xmls = xml_paths_without_test[train_count:]

# TODO : Train/val은 랜덤 셔플을 하기때문에 항상 한세트로 만들어야함. 코드에서 강제 필요
val_annotation_data = generate_segmented_data(val_xmls, drawing_dir, drawing_segment_dir, segment_params, text_xml_dir,
                                              symbol_dict, include_text_as_class, include_text_orientation_as_class, drawing_resize_scale, "val")
write_coco_annotation(os.path.join(drawing_segment_dir,"val.json"), val_annotation_data, symbol_dict, segment_params)
train_annotation_data = generate_segmented_data(train_xmls, drawing_dir, drawing_segment_dir, segment_params, text_xml_dir,
                                                symbol_dict, include_text_as_class, include_text_orientation_as_class, drawing_resize_scale, "train")
write_coco_annotation(os.path.join(drawing_segment_dir,"train.json"), train_annotation_data, symbol_dict, segment_params)

test_annotation_data = generate_segmented_data(test_xmls, drawing_dir, drawing_segment_dir, segment_params, text_xml_dir,
                                               symbol_dict, include_text_as_class, include_text_orientation_as_class, drawing_resize_scale, "test")
write_coco_annotation(os.path.join(drawing_segment_dir,"test.json"), test_annotation_data, symbol_dict, segment_params)