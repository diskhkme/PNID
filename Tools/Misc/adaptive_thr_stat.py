import os
import numpy as np
from Common.pnid_xml import symbol_xml_reader
from Common.symbol_io import read_symbol_txt
from collections import defaultdict

base_dir = "D:/Test_Models/PNID/HyundaiEng/210520_Data/"
drawing_dir = base_dir + "Drawing"
symbol_xml_dir = base_dir + "SymbolXML"
text_xml_dir = base_dir + "TextXML"

val_drawings = ['26071-200-M6-052-00004', '26071-200-M6-052-00013', '26071-200-M6-052-00015', '26071-200-M6-052-00021',
                '26071-200-M6-052-00032', '26071-200-M6-052-00036', '26071-200-M6-052-00048', '26071-200-M6-052-00074',
                '26071-200-M6-052-00081', '26071-200-M6-052-00083', '26071-200-M6-052-00084', '26071-200-M6-052-00086',
                '26071-200-M6-052-00101', '26071-200-M6-052-00115', '26071-300-M6-053-00004', '26071-300-M6-053-00007',
                '26071-300-M6-053-00021', '26071-300-M6-053-00301', '26071-500-M6-059-00021', '26071-500-M6-059-00024']
test_drawings = ['26071-200-M6-052-00002', '26071-200-M6-052-00005', '26071-200-M6-052-00006', '26071-200-M6-052-00056',
                '26071-200-M6-052-00077', '26071-200-M6-052-00107', '26071-200-M6-052-00120', '26071-300-M6-053-00003',
                '26071-300-M6-053-00025', '26071-300-M6-053-00027', '26071-300-M6-053-00263', '26071-300-M6-053-00271',
                '26071-300-M6-053-00302', '26071-300-M6-053-00305', '26071-300-M6-053-00310', '26071-500-M6-059-00007',
                '26071-500-M6-059-00009', '26071-500-M6-059-00014', '26071-500-M6-059-00017', '26071-500-M6-059-00022']
ignore_drawing = []
train_drawings = [x.split(".")[0] for x in os.listdir(symbol_xml_dir)
                  if x.split(".")[0] not in test_drawings and
                  x.split(".")[0] not in val_drawings and
                  x.split(".")[0] not in ignore_drawing ]

symbol_txt_path = base_dir + "Hyundai_SymbolClass_Sym_Only.txt"

include_text_as_class = True # Text를 별도의 클래스로 포함할 것인지 {"text"}
include_text_orientation_as_class = True # 세로 문자열을 또다른 별도의 클래스로 포함할 것인지 {"text_rotated"},
# TODO: 현대ENG 데이터에는 45도 회전 데이터도 있어서 {"text_rotated_45"} 심볼도 추가. 인식 이후 과정에서 문제가 없는지 테스트 필요함

symbol_dict = read_symbol_txt(symbol_txt_path, include_text_as_class, include_text_orientation_as_class)

train_xmls = [os.path.join(symbol_xml_dir, f"{x}.xml") for x in train_drawings]
val_xmls = [os.path.join(symbol_xml_dir, f"{x}.xml") for x in val_drawings]
test_xmls = [os.path.join(symbol_xml_dir, f"{x}.xml") for x in test_drawings]

self_interection_stat = defaultdict(list)
for xml in train_xmls + val_xmls:
    print(xml)
    sym_data = symbol_xml_reader(xml)

    boxes = np.array([[x[1],x[2],x[3],x[4]] for x in sym_data.object_list])
    boxes[:,2] = boxes[:,2] - boxes[:,0]
    boxes[:, 3] = boxes[:, 3] - boxes[:, 1] # [xmin, ymin, xmax, ymax] -> [x, y, w, h]

    classes = np.array([symbol_dict[x[0]] for x in sym_data.object_list])

    unique_classes = set(classes)

    for current_class in unique_classes:
        label_mask = classes == current_class

        x1 = boxes[label_mask, 0]
        y1 = boxes[label_mask, 1]
        w = boxes[label_mask, 2]
        h = boxes[label_mask, 3]
        area = w * h

        for i in range(x1.shape[0]-1):
            # With vector implementation, we can calculate fast
            xx1 = np.maximum(x1[i], x1[i+1:])
            yy1 = np.maximum(y1[i], y1[i+1:])
            xx2 = np.minimum(x1[i] + w[i], x1[i+1:] + w[i+1:])
            yy2 = np.minimum(y1[i] + h[i], y1[i+1:] + h[i+1:])

            w_ = np.maximum(0, xx2 - xx1 + 1)
            h_ = np.maximum(0, yy2 - yy1 + 1)
            intersection = w_ * h_

            # Calculate the iou
            iou = intersection / (area[i+1:] + area[i] - intersection)

            self_intersected = [x for x in iou if x > 0 and x < 1]
            if len(self_intersected) > 0:
                self_interection_stat[current_class].extend(self_intersected)

print(self_interection_stat)

for label, ious in self_interection_stat.items():
    iou_arr = np.array(ious)
    named_label = [k for k,v in symbol_dict.items() if v == label]
    print(f"Label {label}({named_label[0]}), Mean IOU: {np.mean(iou_arr)}, Max IOU: {np.max(iou_arr)}, Min IOU: {np.min(iou_arr)}")