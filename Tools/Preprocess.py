import os
import random
from SegmentImages import segment_images_in_dataset
from convert_symbol import read_symbol_txt, symbol_simple_dump
from write_coco_annotation import write_coco_annotation

drawing_folder = "D:/Test_Models/PNID/EWP_Data/Drawing"
drawing_segment_folder = "D:/Test_Models/PNID/EWP_Data/Drawing_Segment"
symbolXMLFolder = "D:/Test_Models/PNID/EWP_Data/SymbolXML"
testDrawings = ["KNU-A-22300-001-04", "KNU-A-36420-014-03",
                "KNU-A-71710-003-02", "KNU-B-11600-001-03",
                "KNU-B-11600-002-03", "KNU-B-36420-019-04",
                "KNU-B-36610-002-05", "KNU-B-36610-004-01",
                "KNU-B-36610-005-03"]
symbol_id_name = "D:/Test_Models/PNID/EWP_Data/Symbol Class List.pbtxt"
train_ratio = 0.9
segment_params = [800, 800, 400, 400] # width_size, height_size, width_stride, height_stride


symbol_dict = read_symbol_txt(symbol_id_name, start_id=1, merge=False)
symbol_simple_dump("symboldict.txt",symbol_dict)


xml_paths_without_test = [os.path.join(symbolXMLFolder, x) for x in os.listdir(symbolXMLFolder) if x.split(".")[0] not in testDrawings]

random.shuffle(xml_paths_without_test)
train_count = int(len(xml_paths_without_test)*train_ratio)
train_xmls = xml_paths_without_test[0:train_count]
val_xmls = xml_paths_without_test[train_count:]


train_annotation_data = segment_images_in_dataset(train_xmls, drawing_folder, drawing_segment_folder, segment_params)
val_annotation_data = segment_images_in_dataset(val_xmls, drawing_folder, drawing_segment_folder, segment_params)

write_coco_annotation("val_annotation.json", val_annotation_data, symbol_dict, segment_params)
write_coco_annotation("train_annotation.json", train_annotation_data, symbol_dict, segment_params)

