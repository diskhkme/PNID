import cv2
import os
import numpy as np
from Common.pnid_xml import symbol_xml_reader, text_xml_reader
from pathlib import Path


def calculate_diagonal(bbox):
    """
    Args:
        bbox: [int], [xmin, ymin, xmax, ymax]

    Returns: diagonal : int, diagonal length of input bbox
    """
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    return np.sqrt(width ** 2 + height ** 2)


def big_symbol_check(bbox, diagonal_threshold=700):
    """
    Args:
        bbox: [int], [xmin, ymin, xmax, ymax]

    Returns: bool, if diagonal length of bbox longer than the threshold return True else False
    """
    return calculate_diagonal(bbox) > diagonal_threshold


def apply_erode(img, kernel_size=(2, 2)):
    kernel = np.ones(kernel_size, np.uint8)
    erode = cv2.erode(img, kernel, iterations=4)

    return erode


def apply_resize(img, resize_factor):
    height, width, channels = img.shape
    img_resized = cv2.resize(img, (int(width * resize_factor), int(height * resize_factor)))

    return img_resized


def generate_bigsize_data(xml_list, drawing_dir, drawing_output_dir, text_xml_dir, symbol_dict, include_text_as_class, include_text_orientation_as_class,
                            drawing_resize_scale, prefix):
    """ 폴더 내 원본 이미지 도면들을 분할하고 분할 정보를 리스트로 저장

    Arguments:
        xml_list (list): xml 파일 리스트
        drawing_dir (string): 원본 도면 이미지가 존재하는 폴더
        drawing_output_dir (string): 분할된 이미지를 저장할 폴더
        text_xml_dir (string): text xml 파일의 폴더 (include_text_as_calss가 False면 사용하지 않음)
        symbol_dict (dict): symbol 이름을 key로, id를 value로 갖는 dict
        include_text_as_class (bool): text 데이터를 class로 추가할 것인지
        drawing_resize_scale (float): 전체 도면 조정 스케일
        prefix (string): train/val/test 중 하나. 이미지 저장 폴더명 생성에 필요

    Return:
        xml에 있는 전체 도면에서 분할된 도면의 전체 정보 [sub_img_name, symbol_name, xmin, ymin, xmax, ymax]
    """
    entire_big_sym_info = []
    drawing_dir = Path(drawing_dir)
    text_xml_dir = Path(text_xml_dir)
    drawing_output_dir = Path(drawing_output_dir).joinpath(prefix)
    drawing_output_dir.mkdir(parents=True, exist_ok=True)

    for xmlPath in xml_list:
        print(f"Proceccing {xmlPath} in generate_bigsize_data...")
        image_contain_big_sym = False
        xmlPath = Path(xmlPath)
        ext = xmlPath.suffix
        if ext != ".xml":
            continue

        xmlReader = symbol_xml_reader(str(xmlPath.resolve()))
        img_filename, width, height, depth, object_list = xmlReader.getInfo()

        for obj in object_list:
            # obj [class_name_str, bbox] -> [class_number, bbox]
            # EWP 처럼 class_name-direction 같이 생긴경우 이 split에서 방향이 제거됨
            bbox = obj[1:]
            if big_symbol_check(bbox):
                image_contain_big_sym = True
                class_name_str = obj[0].split("-")[0]
                class_number = symbol_dict[class_name_str]
                obj[0] = class_number

                entire_big_sym_info.append([img_filename, *obj])

        if include_text_as_class and text_xml_dir.joinpath(xmlPath.name).exists():
            text_xml_path = str(text_xml_dir.joinpath(xmlPath.name))
            text_xml_reader_obj = text_xml_reader(text_xml_path)
            _, _, _, _, txt_object_list = text_xml_reader_obj.getInfo()
            for text_obj in txt_object_list:
                # text_obj = [text, xmin, ymin, xmax, ymax, direction]
                text_bbox = text_obj[1:5]

                if big_symbol_check(text_bbox):
                    image_contain_big_sym = True
                    entire_big_sym_info.append([img_filename, symbol_dict['text'], *text_bbox])

        if image_contain_big_sym:
            img_file_path = str(drawing_dir.joinpath(img_filename))
            output_file_path = str(drawing_output_dir.joinpath(img_filename))

            img = cv2.imread(img_file_path)
            erode = apply_erode(img)
            resized = apply_resize(erode, drawing_resize_scale)
            # resized_erode = apply_erode(resized)

            cv2.imwrite(output_file_path, resized)

    for big_sym in entire_big_sym_info:
        # big_sym = [image name, class number, xmin, ymin, xmax, ymax]
        big_sym_bbox = big_sym[2:]
        big_sym_bbox_resized = [int(i * drawing_resize_scale) for i in big_sym_bbox]
        big_sym[2:] = big_sym_bbox_resized

    return entire_big_sym_info


if __name__ == '__main__':
    from Common.symbol_io import read_symbol_txt

    base_dir = "D:/Hyondai_Data/210520_Data/"
    drawing_dir = base_dir + "Drawing/JPG"
    drawing_segment_dir = "./test"
    symbol_xml_dir = base_dir + "SymbolXML"
    text_xml_dir = base_dir + "TextXML"

    val_drawings = ['26071-200-M6-052-00004', '26071-200-M6-052-00013', '26071-200-M6-052-00015',
                    '26071-200-M6-052-00021',
                    '26071-200-M6-052-00032', '26071-200-M6-052-00036', '26071-200-M6-052-00048',
                    '26071-200-M6-052-00074',
                    '26071-200-M6-052-00081', '26071-200-M6-052-00083', '26071-200-M6-052-00084',
                    '26071-200-M6-052-00086',
                    '26071-200-M6-052-00101', '26071-200-M6-052-00115', '26071-300-M6-053-00004',
                    '26071-300-M6-053-00007',
                    '26071-300-M6-053-00021', '26071-300-M6-053-00301', '26071-500-M6-059-00021',
                    '26071-500-M6-059-00024']
    test_drawings = ['26071-200-M6-052-00002', '26071-200-M6-052-00005', '26071-200-M6-052-00006',
                     '26071-200-M6-052-00056',
                     '26071-200-M6-052-00077', '26071-200-M6-052-00107', '26071-200-M6-052-00120',
                     '26071-300-M6-053-00003',
                     '26071-300-M6-053-00025', '26071-300-M6-053-00027', '26071-300-M6-053-00263',
                     '26071-300-M6-053-00271',
                     '26071-300-M6-053-00302', '26071-300-M6-053-00305', '26071-300-M6-053-00310',
                     '26071-500-M6-059-00007',
                     '26071-500-M6-059-00009', '26071-500-M6-059-00014', '26071-500-M6-059-00017',
                     '26071-500-M6-059-00022']
    ignore_drawing = []
    train_drawings = [x.split(".")[0] for x in os.listdir(symbol_xml_dir)
                      if x.split(".")[0] not in test_drawings and
                      x.split(".")[0] not in val_drawings and
                      x.split(".")[0] not in ignore_drawing]

    total_drawings = [x.split(".")[0] for x in os.listdir(symbol_xml_dir)]

    symbol_txt_path = base_dir + "Hyundai_SymbolClass_Sym_Only.txt"

    include_text_as_class = False  # Text를 별도의 클래스로 포함할 것인지 {"text"}
    include_text_orientation_as_class = False  # 세로 문자열을 또다른 별도의 클래스로 포함할 것인지 {"text_rotated"},

    segment_params = [1241, 877, 0, 0]  # width_size, height_size, width_stride, height_stride
    drawing_resize_scale = 0.125

    symbol_dict = read_symbol_txt(symbol_txt_path, include_text_as_class, include_text_orientation_as_class)

    train_xmls = [os.path.join(symbol_xml_dir, f"{x}.xml") for x in train_drawings]
    val_xmls = [os.path.join(symbol_xml_dir, f"{x}.xml") for x in val_drawings]
    test_xmls = [os.path.join(symbol_xml_dir, f"{x}.xml") for x in test_drawings]

    total_xmls = [os.path.join(symbol_xml_dir, f"{x}.xml") for x in total_drawings]

    from Data_Generator.write_coco_annotation import write_coco_annotation

    train_annotation_data = generate_bigsize_data(train_xmls, drawing_dir, drawing_segment_dir,
                                                  text_xml_dir, symbol_dict,
                                                  include_text_as_class, include_text_orientation_as_class,
                                                  drawing_resize_scale, 'train')
    write_coco_annotation(os.path.join(drawing_segment_dir, "train.json"), train_annotation_data, symbol_dict,
                          segment_params)


    train_annotation_data = generate_bigsize_data(val_xmls, drawing_dir, drawing_segment_dir,
                                                  text_xml_dir, symbol_dict,
                                                  include_text_as_class, include_text_orientation_as_class,
                                                  drawing_resize_scale, 'val')
    write_coco_annotation(os.path.join(drawing_segment_dir, "val.json"), train_annotation_data, symbol_dict,
                          segment_params)

    train_annotation_data = generate_bigsize_data(test_xmls, drawing_dir, drawing_segment_dir,
                                                  text_xml_dir, symbol_dict,
                                                  include_text_as_class, include_text_orientation_as_class,
                                                  drawing_resize_scale, 'test')
    write_coco_annotation(os.path.join(drawing_segment_dir, "test.json"), train_annotation_data, symbol_dict,
                          segment_params)
