import json
from collections import defaultdict
from itertools import islice
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET


def read_symbol_classes(SymbolClass_txt_file_path):
    class_index = 0
    class_name_to_index_dict = {}
    class_index_to_name_dict = {}

    with open(SymbolClass_txt_file_path, 'r') as f:
        for l in f.readlines():
            _, class_name = l.rstrip().split("|")
            class_name_to_index_dict[class_name] = class_index
            class_index_to_name_dict[class_index] = class_name

            class_index += 1

    return class_name_to_index_dict, class_index_to_name_dict


def read_EWP_xml(xml_path, class_name_to_index_dict):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    objs = []
    for child in root.findall('object'):
        name_tag = child[0]
        bndbox_tag = child[1]

        name = name_tag.text
        bbox = []
        for xy_minmax in bndbox_tag:
            # xmin, ymin, xmax, ymax
            bbox.append(int(xy_minmax.text))
        # xmin_xml, ymin_xml, xmax_xml, ymax_xml = bbox
        # bbox = [ymin_xml, xmin_xml, ymax_xml, xmax_xml]

        # remove obj direction
        name = name.split(sep='-', maxsplit=1)[0]
        class_index = class_name_to_index_dict[name]

        objs.append((class_index, bbox))

    return objs


def get_images_id_to_name_from_GT_json(json_file):
    image_dict_list = json_file['images']
    image_id_to_name_dict = {}

    for image_dict in image_dict_list:
        file_name = image_dict['file_name']
        file_name_without_extension = file_name.split('.')[0]
        true_filename, h, w = file_name_without_extension.split('_')
        id = image_dict['id']

        image_id_to_name_dict[id] = [true_filename, int(h), int(w)]

    return image_id_to_name_dict


def get_images_name_to_id_from_GT_json(json_file):
    image_dict_list = json_file['images']
    image_id_to_name = {}

    for image_dict in image_dict_list:
        file_name = image_dict['file_name']
        file_name_without_extension = file_name.split('.')[0]
        id = image_dict['id']

        image_id_to_name[file_name_without_extension] = id

    return image_id_to_name


def result_json_parse(json_file):
    image_id_to_bbox_dict = defaultdict(list)

    for result_dict in json_file:
        image_id = result_dict['image_id']
        image_id_to_bbox_dict[image_id].append(dict(islice(result_dict.items(), 1, None)))

    return image_id_to_bbox_dict


def bbox_coordinate_convert_grid_to_image(image_id_to_name_dict, image_id_to_bbox_dict, stride_w, stride_h):
    image_id_to_bbox_dict_ = deepcopy(image_id_to_bbox_dict)
    for image_id in image_id_to_bbox_dict_.keys():
        bboxs_info = image_id_to_bbox_dict_[image_id]
        image_name, h, w = image_id_to_name_dict[image_id]

        for bbox_info in bboxs_info:
            bbox_in_grid = bbox_info['bbox']
            bbox_in_grid[0] = int(bbox_in_grid[0] + stride_w * w)
            bbox_in_grid[1] = int(bbox_in_grid[1] + stride_h * h)
            bbox_in_grid[2] = int(bbox_in_grid[2])
            bbox_in_grid[3] = int(bbox_in_grid[3])

    return image_id_to_bbox_dict_


def collect_bbox_to_whole_image(image_id_to_name_dict, image_id_to_bbox_dict):
    image_name_to_bbox_dict = defaultdict(list)

    for image_id in image_id_to_bbox_dict.keys():
        bboxs_info = image_id_to_bbox_dict[image_id]
        image_name, h, w = image_id_to_name_dict[image_id]

        image_name_to_bbox_dict[image_name] += bboxs_info

    return image_name_to_bbox_dict


def bbox_dict_list_filter_by_score(bbox_dict_list, score):
    bbox_dict_list_over_score = [
        i for i in bbox_dict_list if i['score'] > score
    ]

    return bbox_dict_list_over_score


def read_EWP_xml_make_dict(xml_dir_path, SymbolClass_txt_file_path):
    image_name_to_gt_bbox_dict = defaultdict(list)
    xml_dir_path = Path(xml_dir_path)
    class_name_to_index_dict, class_index_to_name_dict = read_symbol_classes(SymbolClass_txt_file_path)
    for xml_path in xml_dir_path.glob('**/*.xml'):
        xml_name = xml_path.stem
        xml_path = xml_path.resolve()

        objs = read_EWP_xml(xml_path, class_name_to_index_dict)
        for obj in objs:
            class_id, bbox = obj
            bbox_dict = {}
            bbox_dict['bbox'] = bbox
            bbox_dict['category_id'] = class_id
            bbox_dict['matched'] = False
            image_name_to_gt_bbox_dict[xml_name].append(bbox_dict)

    return dict(image_name_to_gt_bbox_dict)


def make_whole_image_gt_json(xml_dir_path, SymbolClass_txt_file_path):
    images = []
    annotations = []
    categories = []

    xml_dir_path = Path(xml_dir_path)
    class_name_to_index_dict, class_index_to_name_dict = read_symbol_classes(SymbolClass_txt_file_path)

    for class_name, index in class_name_to_index_dict.items():
        category = {"supercategory": class_name, "id": index, "name": class_name}
        categories.append(category)

    image_id = 1
    object_id = 1
    for xml_path in xml_dir_path.glob('**/*.xml'):
        xml_name = xml_path.stem
        xml_path = xml_path.resolve()

        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        image = {
            "file_name": '{}.jpg'.format(xml_name),
            "id": image_id,
            "width": size[0].text,
            "height": size[0].text,
            "coco_url": 'dummy_words',
            "flickr_url": 'dummy_words',
            "date_captured": '2021-03-17 12:34:56',
            "license": 0
        }
        images.append(image)

        objs = read_EWP_xml(xml_path, class_name_to_index_dict)
        for obj in objs:
            class_index, bbox = obj
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1

            obj = {
                "bbox": [x1, y1, width, height]
                , "category_id": class_index,
                "image_id": image_id, 'id': object_id, 'area': width * height, 'segmentation': [],
                'iscrowd': 0
            }
            annotations.append(obj)
            object_id += 1
        image_id += 1

    json_file = {}
    json_file['annotations'] = annotations
    json_file['images'] = images
    json_file['categories'] = categories

    return json_file


def make_whole_image_result_json(whole_image_gt_json, dataset_json, result_json, score_threshold=0.5):
    image_id_to_name_dict = get_images_id_to_name_from_GT_json(dataset_json)
    image_id_to_bbox_dict = result_json_parse(result_json)
    image_id_to_bbox_in_whole_image_dict = bbox_coordinate_convert_grid_to_image(image_id_to_name_dict, image_id_to_bbox_dict, 300, 300)
    image_name_to_bboxs_dict = collect_bbox_to_whole_image(image_id_to_name_dict, image_id_to_bbox_in_whole_image_dict)

    annotations = []

    images_name_to_id_in_whole_img = get_images_name_to_id_from_GT_json(whole_image_gt_json)
    for image_name, bboxs in image_name_to_bboxs_dict.items():
        id = images_name_to_id_in_whole_img[image_name]

        for bbox in bboxs:
            if bbox['score'] > score_threshold:
                bbox['image_id'] = id
                annotations.append(bbox)

    return annotations


def test_code():
    with open('./dataset_0_test.json') as f:
        gt_json = json.load(f)

    with open('./rivision_experiment_1_result.bbox.json') as f:
        result_json = json.load(f)

    image_id_to_name_dict = get_images_id_to_name_from_GT_json(gt_json)
    image_id_to_bbox_dict = result_json_parse(result_json)
    image_id_to_bbox_in_image_dict = bbox_coordinate_convert_grid_to_image(image_id_to_name_dict, image_id_to_bbox_dict, 300, 300)

    image_name_to_bbox_dict = collect_bbox_to_whole_image(image_id_to_name_dict, image_id_to_bbox_in_image_dict)

    for i in image_name_to_bbox_dict.items():
        print(i)


if __name__ == '__main__':
    # test_code()
    #
    # image_name_to_gt_bbox_dict = read_EWP_xml_make_dict('../0. raw_data/original/EWP_Test/xml', '../0. raw_data/EWP_SymbolClass_sym_only.txt')
    #
    # for i in image_name_to_gt_bbox_dict.items():
    #     print(i)

    # whole_dataset_json = make_whole_image_gt_json('../0. raw_data/resized_0.5_add_text/EWP_Test/xml',
    #                                               '../0. raw_data/EWP_SymbolClass_sym_only_plus_txex_class.txt')
    #
    # with open('./{}.json'.format('dataset_3_test_whole_image'), 'w', encoding="utf-8") as f:
    #     json.dump(whole_dataset_json, f, ensure_ascii=False, indent=2)
    #
    # with open('./dataset_3_test.json', 'r') as f:
    #     dataset_json = json.load(f)
    #
    # with open('./dataset_3_sgd.bbox.json', 'r') as f:
    #     result_json = json.load(f)
    with open('./whole_image_json/gt_json/dataset_0_test_whole_image.json', 'r') as f:
        whole_dataset_json = json.load(f)
    with open('./gt_json/dataset_0_test.json', 'r') as f:
        dataset_json = json.load(f)
    with open('./aaaa_dataset0.bbox.json', 'r') as f:
        result_json = json.load(f)

    json_file = make_whole_image_result_json(whole_dataset_json, dataset_json, result_json)

    with open('./{}.json'.format('aaaa_dataset0_whole.bbox'), 'w', encoding="utf-8") as f:
        json.dump(json_file, f, ensure_ascii=False, indent=2)
