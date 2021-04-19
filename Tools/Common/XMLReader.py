from xml.etree.ElementTree import parse
import xml.etree.ElementTree as ET
import math
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

class XMLReader():
    def __init__(self, filepath):
        self.filepath = filepath
        self.tree = parse(filepath)
        self.root = self.tree.getroot()

        self.filename = self.root.findtext("filename")
        self.width = int(self.root.find("size").findtext("width"))
        self.height = int(self.root.find("size").findtext("height"))
        self.depth = int(self.root.find("size").findtext("depth"))

        self.objectList = []

    def getInfo(self):
        return self.filename, self.width, self.height, self.depth, self.objectList

class SymbolXMLReader(XMLReader):
    def __init__(self,filepath):
        super().__init__(filepath)

        for object in self.root.iter("object"):
            xmin = int(object.find("bndbox").findtext("xmin"))
            xmax = int(object.find("bndbox").findtext("xmax"))
            ymin = int(object.find("bndbox").findtext("ymin"))
            ymax = int(object.find("bndbox").findtext("ymax"))
            name = object.findtext("name")
            self.objectList.append([name, xmin, ymin, xmax, ymax])

class TextXMLReader(XMLReader):
    def __init__(self,filepath):
        super().__init__(filepath)

        for object in self.root.iter("object"):
            xmin = int(object.find("bndbox").findtext("xmin"))
            xmax = int(object.find("bndbox").findtext("xmax"))
            ymin = int(object.find("bndbox").findtext("ymin"))
            ymax = int(object.find("bndbox").findtext("ymax"))
            string = object.findtext("string")
            orientation = int(math.ceil(float(object.findtext("orientation")))) # 89.9991이 있음 (예외)
            self.objectList.append([string, xmin, ymin, xmax, ymax, orientation])

    def getInfo(self):
        return self.filename, self.width, self.height, self.depth, self.objectList

    def error_correction(self, img_dir, remove_spacing = True, newline_separation = True, remove_blank_pixel = True, remove_blank_threshold = 0.7, margin=5 ):
        obj_to_remove = []
        for object in self.root.iter("object"):
            filename_tag = object.findtext("filename")
            xmin = int(object.find("bndbox").findtext("xmin"))
            xmax = int(object.find("bndbox").findtext("xmax"))
            ymin = int(object.find("bndbox").findtext("ymin"))
            ymax = int(object.find("bndbox").findtext("ymax"))
            string = object.findtext("string")
            orientation = int(math.ceil(float(object.findtext("orientation"))))

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

        for obj in obj_to_remove:
            self.root.remove(obj)

        if remove_blank_pixel == True:
            img_filename = os.path.basename(self.filepath).split(".")[0] + ".jpg"
            img = cv2.imread(os.path.join(img_dir, img_filename), cv2.IMREAD_GRAYSCALE)
            dilated_img = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=1)
            dilated_img = 255 - dilated_img

            for object in self.root.iter("object"):
                filename_tag = object.findtext("filename")
                xmin = int(object.find("bndbox").findtext("xmin"))
                xmax = int(object.find("bndbox").findtext("xmax"))
                ymin = int(object.find("bndbox").findtext("ymin"))
                ymax = int(object.find("bndbox").findtext("ymax"))
                string = object.findtext("string")
                orientation = int(math.ceil(float(object.findtext("orientation"))))

                sub_img = dilated_img[ymin:ymax, xmin:xmax]
                if orientation == 0:
                    pixel_sum_along_height = np.sum(sub_img, axis=0)
                elif orientation == 90:
                    pixel_sum_along_height = np.sum(sub_img, axis=1)

                try:
                    first = np.min(np.nonzero(pixel_sum_along_height))
                    last = np.max(np.nonzero(pixel_sum_along_height))
                except:
                    continue

                assert(first < last, "something wrong...")

                for i in range(len(pixel_sum_along_height)):
                    if pixel_sum_along_height[i] != 0:
                        first = i
                    if pixel_sum_along_height[len(pixel_sum_along_height) - 1 - i] != 0:
                        last = len(pixel_sum_along_height) - 1 - i
                    if first != -1 and last != -1:
                        break

                if string == "V576":
                    k = 1

                # 명시된 크기의 thresold 비율보다도 실제 픽셀이 적을 경우
                if orientation == 0:
                    if (xmax - xmin) * remove_blank_threshold > (last - first):
                        object.find("bndbox").find("xmin").text = str(xmin + first - margin)
                        object.find("bndbox").find("xmax").text = str(xmin + last + margin)
                elif orientation == 90:
                    if (ymax - ymin) * remove_blank_threshold > (last - first):
                        object.find("bndbox").find("ymin").text = str(ymin + first - margin)
                        object.find("bndbox").find("ymax").text = str(ymin + last + margin)

    def write_xml(self, out_filename):
        self.indent(self.tree.getroot())
        self.tree.write(out_filename)

    def indent(self, elem, level=0):  # 자료 출처 https://goo.gl/J8VoDK
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.indent(elem, level + 1)
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
    xmlReader = TextXMLReader(filepath)
    filename, width, height, depth, objectList = xmlReader.getInfo()
    xmlReader.error_correction(img_dir)

    xmlReader.write_xml("text_edit.xml")