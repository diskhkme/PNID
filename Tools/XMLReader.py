from xml.etree.ElementTree import parse

class XMLReader():
    def __init__(self,filepath):
        tree = parse(filepath)
        root = tree.getroot()

        self.filename = root.findtext("filename")
        self.width = int(root.find("size").findtext("width"))
        self.height = int(root.find("size").findtext("height"))
        self.depth = int(root.find("size").findtext("depth"))

        objectList = []
        for object in root.iter("object"):
            xmin = int(object.find("bndbox").findtext("xmin"))
            xmax = int(object.find("bndbox").findtext("xmax"))
            ymin = int(object.find("bndbox").findtext("ymin"))
            ymax = int(object.find("bndbox").findtext("ymax"))
            name = object.findtext("name")
            objectList.append([name, xmin, ymin, xmax, ymax])

        self.objectList = objectList

    def getInfo(self):
        return self.filename, self.width, self.height, self.depth, self.objectList



if __name__ == '__main__':
    filepath = "D:/Test_Models/PNID/EWP_Data/SymbolXML/KNU-A-22300-001-01.xml"
    xmlReader = XMLReader(filepath)
    filename, width, height, depth, objectList = xmlReader.getInfo()
    print(filename, width, height, depth)
    print(objectList)