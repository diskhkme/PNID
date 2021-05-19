import os
import cv2
from Common.pnid_xml import text_xml_reader, symbol_xml_reader
from Visualize.image_drawing import draw_bbox_from_bbox_list

# 현대엔지니어링 이미지 도면에 -001-001이 모두 일괄적으로 저장되어 있어서 이를 없애도록 변경

xml_dir = "D:/Test_Models/PNID/HyundaiEng/210518_Data/Symbol_XML"
drawing_img_dir = "D:/Test_Models/PNID/HyundaiEng/210518_Data/Drawing/JPG_tmp"
output_img_dir = "D:/Test_Models/PNID/HyundaiEng/210518_Data/Symbol_XML_Visualize"
is_text_xml = False

drawing_filenames = os.listdir(drawing_img_dir)

for drawing_filename in drawing_filenames:
    print(drawing_filename)
    name_only = drawing_filename.split(".")[0]
    splitted_name = name_only.split("-")
    corrected_name = ""
    for i in range(len(splitted_name)-2): # 끝의 001 두개를 없애야 함
        corrected_name += splitted_name[i]
        corrected_name += "-"

    corrected_name = corrected_name[:-1]

    from_filename = os.path.join(drawing_img_dir, drawing_filename)
    to_filename = os.path.join(drawing_img_dir, corrected_name + ".jpg")

    os.rename(from_filename, to_filename)

