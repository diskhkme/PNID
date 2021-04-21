import os
import random
import pickle
from Common.xml_reader import text_xml_reader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

# Text 오류 수정 결과 확인을 위한 코드

text_xml_dir = "D:/Test_Models/PNID/EWP_Data/TextXML_Corrected/"
drawing_img_dir = "D:/Test_Models/PNID/EWP_Data/Drawing/"
target_drawing_name = "KNU-B-36130-001-03"

text_xml_path = os.path.join(text_xml_dir, target_drawing_name + ".xml")
drawing_path = os.path.join(drawing_img_dir, target_drawing_name + ".jpg")

text_xml_reader = text_xml_reader(text_xml_path)
filename, width, height, depth, objectList = text_xml_reader.getInfo()

img = cv2.imread(drawing_path)
for object in objectList:
    cv2.rectangle(img,(object[1], object[2]),
                  (object[3], object[4]),
                  (255,0,0),2)
    cv2.putText(img,object[0],(object[1],object[2]), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0))

cv2.imwrite(f"{target_drawing_name}_corr_viz.png",img)