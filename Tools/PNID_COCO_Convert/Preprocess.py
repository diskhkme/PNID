import os
import random
import pickle
from PNID_COCO_Convert.SegmentImages import segment_images_in_dataset
from PNID_COCO_Convert.convert_symbol import read_symbol_txt, symbol_simple_dump
from PNID_COCO_Convert.write_coco_annotation import write_coco_annotation

base_dir = "D:/Test_Models/PNID/EWP_Data/"
drawing_folder = base_dir + "Drawing"
drawing_segment_folder = base_dir + "Drawing_Segment"
symbol_xml_folder = base_dir + "SymbolXML"
text_xml_folder = base_dir + "TextXML"
test_drawings = ["KNU-A-22300-001-04", "KNU-A-36420-014-03",
                "KNU-A-71710-003-02", "KNU-B-11600-001-03",
                "KNU-B-11600-002-03", "KNU-B-36420-019-04",
                "KNU-B-36610-002-05", "KNU-B-36610-004-01",
                "KNU-B-36610-005-03"]
symbol_id_name = base_dir + "Symbol Class List.pbtxt"

include_text_as_class = False
train_ratio = 0.9
segment_params = [800, 800, 300, 300] # width_size, height_size, width_stride, height_stride
start_id = 1

symbol_dict = read_symbol_txt(symbol_id_name, start_id=start_id, merge=True)
if include_text_as_class == True:
    symbol_dict["text"] = len(symbol_dict.items()) + start_id

symbol_simple_dump("symboldict.txt", symbol_dict)

xml_paths_without_test = [os.path.join(symbol_xml_folder, x) for x in os.listdir(symbol_xml_folder) if x.split(".")[0] not in test_drawings]
test_xmls = [os.path.join(symbol_xml_folder, f"{x}.xml") for x in test_drawings]

random.shuffle(xml_paths_without_test)
train_count = int(len(xml_paths_without_test)*train_ratio)
train_xmls = xml_paths_without_test[0:train_count]
val_xmls = xml_paths_without_test[train_count:]


# train_annotation_data = segment_images_in_dataset(train_xmls, drawing_folder, drawing_segment_folder, segment_params, text_xml_folder, include_text_as_class)
# val_annotation_data = segment_images_in_dataset(val_xmls, drawing_folder, drawing_segment_folder, segment_params, text_xml_folder, include_text_as_class)
test_annotation_data = segment_images_in_dataset(test_xmls, drawing_folder, drawing_segment_folder, segment_params, text_xml_folder, include_text_as_class)


# write_coco_annotation("val_annotation.json", val_annotation_data, symbol_dict, segment_params)
# write_coco_annotation("train_annotation.json", train_annotation_data, symbol_dict, segment_params)
test_img_dict = write_coco_annotation("test_annotation.json", test_annotation_data, symbol_dict, segment_params)

with open("test_img_dict.pickle", "wb") as f:
    pickle.dump(test_img_dict, f)

