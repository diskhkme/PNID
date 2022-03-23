import os
import random
import pickle
from Common.pnid_xml import text_xml_reader, symbol_xml_reader

# Text XML에 존재하는 오류(앞뒤공백, multiline 박스 분할, 박스 크기 조정, 오류가 있는 object 제거)를 수정하여 다시 XML로 출력

source_xml_dir = "D:/Test_Models/PNID/HyundaiEng/210520_Data/_SymbolXML_before_correction"
drawing_img_dir = "D:/Test_Models/PNID/HyundaiEng/210520_Data/Drawing/JPG"
processed_xml_dir = "D:/Test_Models/PNID/HyundaiEng/210520_Data/SymbolXML/"
is_text_xml = False

xmls = [os.path.join(source_xml_dir, x) for x in os.listdir(source_xml_dir)]

for xml in xmls:
    print(xml)

    if is_text_xml == True:
        xml_obj = text_xml_reader(xml)
        xml_obj.error_correction(drawing_img_dir, remove_spacing = True, newline_separation = True, remove_blank_pixel = False,
                                      remove_blank_threshold = 0.7, margin=5,
                                      remove_none_string = True, remove_object_out_of_img = True)
    else:
        xml_obj = symbol_xml_reader(xml)
        xml_obj.error_correction(drawing_img_dir, remove_empty_object=True, remove_object_out_of_img=True)

    filename = os.path.basename(xml)
    xml_obj.write_xml(os.path.join(processed_xml_dir, filename))


