import os
import random
import pickle
from Common.pnid_xml import text_xml_reader

# Text XML에 존재하는 오류(앞뒤공백, multiline 박스 분할, 박스 크기 조정, 오류가 있는 object 제거)를 수정하여 다시 XML로 출력

text_xml_dir = "D:/Test_Models/PNID/HyundaiEng/210520_Data/_Text_XML_before_correction"
drawing_img_dir = "D:/Test_Models/PNID/HyundaiEng/210520_Data/Drawing/JPG"
processed_text_xml_dir = "D:/Test_Models/PNID/HyundaiEng/210520_Data/TextXML/"

text_xmls = [os.path.join(text_xml_dir, x) for x in os.listdir(text_xml_dir)]

for text_xml in text_xmls:
    print(text_xml)
    text_xml_obj = text_xml_reader(text_xml)
    text_xml_obj.error_correction(drawing_img_dir, remove_spacing = True, newline_separation = True, remove_blank_pixel = False,
                                  remove_blank_threshold = 0.7, margin=5,
                                  remove_none_string = True, remove_object_out_of_img = True)

    filename = os.path.basename(text_xml)
    text_xml_obj.write_xml(os.path.join(processed_text_xml_dir, filename))


