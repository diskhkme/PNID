import os
import cv2
from Common.pnid_xml import text_xml_reader, symbol_xml_reader
from Visualize.image_drawing import draw_bbox_from_bbox_list

# XML 데이터 검증을 위한 가시화 코드

xml_dir = "D:/Test_Models/PNID/HyundaiEng/210518_Data/Text_XML"
drawing_img_dir = "D:/Test_Models/PNID/HyundaiEng/210518_Data/Drawing/JPG"
output_img_dir = "D:/Test_Models/PNID/HyundaiEng/210518_Data/Text_XML_Visualize"
is_text_xml = True

xml_filenames = os.listdir(xml_dir)

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

    img = cv2.imread(drawing_path)

    bbox = [[x[1], x[2], x[3]-x[1], x[4]-x[2]] for x in object_list]
    bbox_data = [x[0] for x in object_list]

    draw_img = draw_bbox_from_bbox_list(img,bbox,bbox_data,(255,0,0),thickness=2)

    if is_text_xml == True:
        out_path = os.path.join(output_img_dir, f"{name_only}_text" + ".jpg")
    else:
        out_path = os.path.join(output_img_dir, f"{name_only}_symbol" + ".jpg")

    cv2.imwrite(out_path,draw_img)