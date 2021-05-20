import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from Common.pnid_xml import text_xml_reader, symbol_xml_reader
from Visualize.image_drawing import draw_bbox_from_bbox_list

# XML 데이터 검증을 위한 가시화 코드

xml_dir = "D:/Test_Models/PNID/HyundaiEng/210518_Data/Symbol_XML"
drawing_img_dir = "D:/Test_Models/PNID/HyundaiEng/210518_Data/Drawing/JPG"
is_text_xml = False

xml_filenames = os.listdir(xml_dir)

entire_objects = []

for xml_filename in xml_filenames:
    print(xml_filename)
    name_only = xml_filename.split(".")[0]
    xml_path = os.path.join(xml_dir, name_only + ".xml")
    drawing_path = os.path.join(drawing_img_dir, name_only + ".jpg")

    if is_text_xml == True:
        text_xml_reader_obj = text_xml_reader(xml_path)
        filename, width, height, depth, object_list = text_xml_reader_obj.getInfo()
    else:
        symbol_xml_reader_obj = symbol_xml_reader(xml_path)
        filename, width, height, depth, object_list = symbol_xml_reader_obj.getInfo()

    bbox = [[x[0], x[1], x[2], x[3]-x[1], x[4]-x[2]] for x in object_list]
    entire_objects.extend(bbox)

unique_labels = set([x[0] for x in entire_objects])
occurences = {}
mean_diagonal_lengths = {}
for unique_label in unique_labels:
    current_lables = [x[0] for x in entire_objects if x[0] == unique_label]
    occurences[unique_label] = len(current_lables)
    current_bboxes = [[x[1],x[2],x[3],x[4]] for x in entire_objects if x[0] == unique_label]
    current_bboxes_array = np.array(current_bboxes)
    mean_diagonal_lengths[unique_label] = np.mean(np.sqrt(current_bboxes_array[:,2] ** 2 + current_bboxes_array[:,3] ** 2))


print(f"Number of symbol instances in entire dataset: {len(entire_objects)}")
print(f"Number of unique symbol labels in entire dataset: {len(unique_labels)}")
entire_bboxes_array = np.array([[x[1], x[2], x[3], x[4]] for x in entire_objects])
sorted_by_size = sorted(mean_diagonal_lengths.items(), key=lambda item: item[1])

f = open("symbol_statistics.csv",'w')
for size in sorted_by_size:
    data = str(size[0]) + "," + str(size[1]) + "," + str(occurences[size[0]]) + "\n"
    f.write(data)
f.close()