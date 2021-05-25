from xml.etree.ElementTree import parse
import xml.etree.ElementTree as ET
import math
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from xml.etree.ElementTree import Element, ElementTree, dump

def write_symbol_result_to_xml(out_dir, dt_result, symbol_dict, symbol_type_dict=None):
    for filename, objects in dt_result.items():
        root = Element("annotation")
        filename_node = Element("filename")
        filename_node.text = f"{filename}.jpg"
        root.append(filename_node)
        for object in objects:

            # text가 key에 있는경우 제외 (텍스트는 별도 XML로 출력함)
            if "text" in symbol_dict:
                if object["category_id"] == symbol_dict["text"] or object["category_id"] == symbol_dict["text_rotated"] or object["category_id"] == symbol_dict["text_rotated_45"]:
                    continue

            object_node = Element("symbol_object")

            name_node = Element("class")
            symbol_name = [sym_name for sym_name, id in symbol_dict.items() if id == object["category_id"]][0]
            name_node.text = symbol_name

            if symbol_type_dict is not None:
                type_node = Element("type")
                type_node.text = [typename_name for sym_name, typename_name in symbol_type_dict.items() if sym_name == symbol_name][0]

            bndbox_node = Element("bndbox")

            xmin_node = Element("xmin")
            xmin_node.text = str(object["bbox"][0])
            ymin_node = Element("ymin")
            ymin_node.text = str(object["bbox"][1])
            xmax_node = Element("xmax")
            xmax_node.text = str(object["bbox"][0] + object["bbox"][2])
            ymax_node = Element("ymax")
            ymax_node.text = str(object["bbox"][1] + object["bbox"][3])

            bndbox_node.append(xmin_node)
            bndbox_node.append(ymin_node)
            bndbox_node.append(xmax_node)
            bndbox_node.append(ymax_node)

            degree_node = Element("degree")
            degree_node.text = str(0)

            flip_node = Element("flip")
            flip_node.text = "n"

            etc_node = Element("etc")
            etc_node.text = ""

            object_node.append(type_node)
            object_node.append(name_node)
            object_node.append(bndbox_node)
            object_node.append(degree_node)
            object_node.append(flip_node)
            object_node.append(etc_node)

            root.append(object_node)

        indent(root)
        out_path = os.path.join(out_dir, f"{filename}.xml")
        ElementTree(root).write(out_path)

def write_text_result_to_xml(out_dir, dt_result_text, symbol_dict):
    for filename, objects in dt_result_text.items():
        root = Element("annotation")
        filename_node = Element("filename")
        filename_node.text = f"{filename}.jpg"
        root.append(filename_node)
        for object in objects:

            object_node = Element("object")

            string_node = Element("string")
            string_node.text = object["string"]

            orientation_node = Element("orientation")
            if object["category_id"] == symbol_dict["text_rotated"]:
                orientation_node.text = str(90)
            elif object["category_id"] == symbol_dict["text_rotated_45"]:
                orientation_node.text = str(45)
            else:
                orientation_node.text = str(0)

            bndbox_node = Element("bndbox")

            xmin_node = Element("xmin")
            xmin_node.text = str(object["bbox"][0])
            ymin_node = Element("ymin")
            ymin_node.text = str(object["bbox"][1])
            xmax_node = Element("xmax")
            xmax_node.text = str(object["bbox"][0] + object["bbox"][2])
            ymax_node = Element("ymax")
            ymax_node.text = str(object["bbox"][1] + object["bbox"][3])

            bndbox_node.append(xmin_node)
            bndbox_node.append(ymin_node)
            bndbox_node.append(xmax_node)
            bndbox_node.append(ymax_node)

            object_node.append(string_node)
            object_node.append(orientation_node)
            object_node.append(bndbox_node)
            root.append(object_node)

        indent(root)
        out_path = os.path.join(out_dir, f"{filename}_text.xml")
        ElementTree(root).write(out_path)


class xml_reader():
    """
    도면 인식용 심볼 및 XML 파일 파싱 기본 클래스

    Arguments:
        string filepath : xml 파일 경로
        xml.etree.ElementTree tree : xml Element tree
        xml.etree.Element root : xml root element
        string filename : xml파일의 정보가 해당하는 도면의 이름
        int width, height, depth : 도면 이미지의 해상도 및 채널
        dict object_list : xml 내 object 정보를 저장하는 list
                        (심볼의 경우 [symbolname, xmin, ymin, xmax, ymax])
                        (텍스트의 경우 [text, xmin, ymin, xmax, ymax, orientation])

    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.tree = parse(filepath)
        self.root = self.tree.getroot()

        self.filename = self.root.findtext("filename")
        self.width = int(self.root.find("size").findtext("width"))
        self.height = int(self.root.find("size").findtext("height"))
        self.depth = int(self.root.find("size").findtext("depth"))

        self.object_list = []

    def getInfo(self):
        return self.filename, self.width, self.height, self.depth, self.object_list

    def write_xml(self, out_filename):
        indent(self.tree.getroot())
        self.tree.write(out_filename)

class symbol_xml_reader(xml_reader):
    """
    심볼 xml 파일 파싱 클래스
    """
    def __init__(self,filepath):
        super().__init__(filepath)

        for object in self.root.iter("object"):
            xmin = int(object.find("bndbox").findtext("xmin"))
            xmax = int(object.find("bndbox").findtext("xmax"))
            ymin = int(object.find("bndbox").findtext("ymin"))
            ymax = int(object.find("bndbox").findtext("ymax"))
            name = object.findtext("name")
            self.object_list.append([name, xmin, ymin, xmax, ymax])

    def error_correction(self, img_dir, remove_empty_object=True, remove_object_out_of_img=True):
        """
        심볼 xml관련하여 존재하는 오차들을 수정하기 위한 클래스 메소드

        Arguments:
            img_dir (string): 도면 이미지가 저장되어 있는 폴더. remove_blank_pixel이 true일때 사용
            remove_empty_object (bool): 빈 객체의 경우 삭제할지 여부
            remove_object_out_of_img (bool) : box 좌표가 도면 밖의 위치일 때 (ex : 좌표값이 음수) 해당 object를 제거할 것인지 여부

        Return:
            None (self.object_list의 박스 좌표가 변화함)
        """

        filename, width, height, depth, object_list = self.getInfo()

        obj_to_remove = []
        for object in self.root.iter("object"):
            filename_tag = object.findtext("filename")
            try:
                xmin = int(object.find("bndbox").findtext("xmin"))
                xmax = int(object.find("bndbox").findtext("xmax"))
                ymin = int(object.find("bndbox").findtext("ymin"))
                ymax = int(object.find("bndbox").findtext("ymax"))
            except:
                obj_to_remove.append(object)
                continue

            symbol_name = object.findtext("name")

            if remove_empty_object == True:
                if symbol_name == "":
                    obj_to_remove.append(object)

            if remove_object_out_of_img == True:
                if xmin < 0 or ymin < 0 or xmax >= width or ymax >= height:
                    obj_to_remove.append(object)

        obj_to_remove = set(obj_to_remove)
        for obj in obj_to_remove:
            self.root.remove(obj)

class text_xml_reader(xml_reader):
    """
    텍스트 xml 파일 파싱 클래스
    """
    def __init__(self,filepath):
        super().__init__(filepath)

        for object in self.root.iter("object"):
            xmin = int(object.find("bndbox").findtext("xmin"))
            xmax = int(object.find("bndbox").findtext("xmax"))
            ymin = int(object.find("bndbox").findtext("ymin"))
            ymax = int(object.find("bndbox").findtext("ymax"))
            string = object.findtext("string")
            orientation = int(math.ceil(float(object.findtext("orientation")))) # 89.9991이 있음 (예외)
            self.object_list.append([string, xmin, ymin, xmax, ymax, orientation])

    def error_correction(self, img_dir, remove_spacing=True, newline_separation=True,
                         remove_blank_pixel=True, remove_blank_threshold=0.7, margin=5,
                         remove_none_string=True, remove_object_out_of_img=True):
        """
        텍스트 xml관련하여 존재하는 오차들을 수정하기 위한 클래스 메소드

        Arguments:
            img_dir (string): 도면 이미지가 저장되어 있는 폴더. remove_blank_pixel이 true일때 사용
            remove_spacing (bool): 문자열 앞뒤의 공백을 trim할 것인지 여부
            newline_separation (bool): 멀티라인 문자의 경우 \n을 기반으로 박스를 분할할 것인지 여부
            remove_blank_pixel (bool): 박스가 문자열보다 지나치게 크게 설정된 경우 인식하여 박스를 줄일 것인지 여부
            remove_blank_threshold (float): 박스 길이 * threshold > 문자열 픽셀 길이일 경우 축소 수행
            margin (int): 축소한 뒤 margin만큼 박스 길이를 늘림
            remove_none_string (bool) : string 정보가 없을 때 해당 object를 제거할 것인지 여부
            remove_object_out_of_img (bool) : box 좌표가 도면 밖의 위치일 때 (ex : 좌표값이 음수) 해당 object를 제거할 것인지 여부

        Return:
            None (self.object_list에서 제거되야 할 object들이 삭제됨)
        """

        filename, width, height, depth, object_list = self.getInfo()

        obj_to_remove = []
        for object in self.root.iter("object"):
            filename_tag = object.findtext("filename")
            try:
                xmin = int(object.find("bndbox").findtext("xmin"))
                xmax = int(object.find("bndbox").findtext("xmax"))
                ymin = int(object.find("bndbox").findtext("ymin"))
                ymax = int(object.find("bndbox").findtext("ymax"))
            except:
                obj_to_remove.append(object)
                continue

            string = object.findtext("string")
            orientation = int(math.ceil(float(object.findtext("orientation"))))

            if remove_none_string == True:
                if string == "":
                    obj_to_remove.append(object)

            if remove_object_out_of_img == True:
                if xmin < 0 or ymin < 0 or xmax >= width or ymax >= height:
                    obj_to_remove.append(object)

            if remove_spacing == True:
                object.find("string").text = string.strip()

            if newline_separation == True:
                if string.find("\n") != -1:
                    strs = string.split("\n")
                    if orientation == 0:
                        line_height = math.floor((ymax-ymin)/len(strs))
                    elif orientation == 90:
                        line_height = math.floor((xmax - xmin) / len(strs))

                    for i in range(len(strs)):
                        obj = ET.SubElement(self.root, "object")
                        ET.SubElement(obj, "filename").text = filename_tag
                        ET.SubElement(obj, "string").text = strs[i]
                        ET.SubElement(obj, "orientation").text = str(orientation)
                        bnd = ET.SubElement(obj, "bndbox")
                        if orientation == 0:
                            ET.SubElement(bnd, "xmin").text = str(xmin)
                            ET.SubElement(bnd, "ymin").text = str(ymin + i*line_height)
                            ET.SubElement(bnd, "xmax").text = str(xmax)
                            ET.SubElement(bnd, "ymax").text = str(ymin + (i+1)*line_height)
                        elif orientation == 90:
                            ET.SubElement(bnd, "xmin").text = str(xmin + i * line_height)
                            ET.SubElement(bnd, "ymin").text = str(ymin)
                            ET.SubElement(bnd, "xmax").text = str(xmin + (i + 1) * line_height)
                            ET.SubElement(bnd, "ymax").text = str(ymax)

                    obj_to_remove.append(object)

        obj_to_remove = set(obj_to_remove)
        for obj in obj_to_remove:
            self.root.remove(obj)

        if remove_blank_pixel == True:
            img_filename = os.path.basename(self.filepath).split(".")[0] + ".jpg"
            img = cv2.imread(os.path.join(img_dir, img_filename), cv2.IMREAD_GRAYSCALE)
            dilated_img = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=1)
            dilated_img = 255 - dilated_img

            for object in self.root.iter("object"):
                filename_tag = object.findtext("filename")
                try:
                    xmin = int(object.find("bndbox").findtext("xmin"))
                    xmax = int(object.find("bndbox").findtext("xmax"))
                    ymin = int(object.find("bndbox").findtext("ymin"))
                    ymax = int(object.find("bndbox").findtext("ymax"))
                except:
                    continue

                string = object.findtext("string")
                orientation = int(math.ceil(float(object.findtext("orientation"))))

                sub_img = dilated_img[ymin:ymax, xmin:xmax]
                if orientation == 0:
                    pixel_sum_along_height = np.sum(sub_img, axis=0)
                    first_original_value = xmin
                    last_original_value = xmax
                elif orientation == 90:
                    pixel_sum_along_height = np.sum(sub_img, axis=1)
                    first_original_value = ymin
                    last_original_value = ymax
                try:
                    first = np.min(np.nonzero(pixel_sum_along_height))
                    last = np.max(np.nonzero(pixel_sum_along_height))
                except:
                    continue

                # if (first < last):
                #     # 텍스트 박스인데 안에 픽셀이 없다는 의미? 그냥 원래대로 복구
                #     first = first_original_value
                #     last = last_original_value


                for i in range(len(pixel_sum_along_height)):
                    if pixel_sum_along_height[i] != 0:
                        first = i
                    if pixel_sum_along_height[len(pixel_sum_along_height) - 1 - i] != 0:
                        last = len(pixel_sum_along_height) - 1 - i
                    if first != -1 and last != -1:
                        break

                # 명시된 크기의 thresold 비율보다도 실제 픽셀이 적을 경우
                if orientation == 0:
                    if (xmax - xmin) * remove_blank_threshold > (last - first):
                        object.find("bndbox").find("xmin").text = str(xmin + first - margin)
                        object.find("bndbox").find("xmax").text = str(xmin + last + margin)
                elif orientation == 90:
                    if (ymax - ymin) * remove_blank_threshold > (last - first):
                        object.find("bndbox").find("ymin").text = str(ymin + first - margin)
                        object.find("bndbox").find("ymax").text = str(ymin + last + margin)

def indent(elem, level=0):  # 자료 출처 https://goo.gl/J8VoDK
    """ XML의 들여쓰기 포함한 출력을 위한 함수

    """
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

if __name__ == '__main__':
    # filepath = "D:/Test_Models/PNID/EWP_Data/SymbolXML/KNU-A-22300-001-01.xml"
    # xmlReader = SymbolXMLReader(filepath)
    # filename, width, height, depth, objectList = xmlReader.getInfo()
    # print(filename, width, height, depth)
    # print(objectList)

    filepath = "D:/Test_Models/PNID/EWP_Data/TextXML/KNU-B-36130-001-03.xml"
    img_dir = "D:/Test_Models/PNID/EWP_Data/Drawing/"
    xmlReader = text_xml_reader(filepath)
    filename, width, height, depth, objectList = xmlReader.getInfo()
    xmlReader.error_correction(img_dir)

    xmlReader.write_xml("text_edit.xml")