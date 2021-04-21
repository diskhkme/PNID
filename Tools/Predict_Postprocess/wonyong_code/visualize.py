import json
from pathlib import Path
import cv2
from collections import defaultdict
import numpy as np


def draw_bbox_from_whole_img_gt_json(image_dir_path, output_dir, json_file, color=(0, 0, 0), thickness=3):
    image_dir_path = Path(image_dir_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    bboxs = json_file['annotations']
    images_info = json_file['images']

    image_name_bboxs_dict = defaultdict(list)
    image_id_name_dict = {}

    for image_info in images_info:
        file_name = image_info['file_name']
        id = image_info['id']
        image_id_name_dict[id] = file_name

    for bbox in bboxs:
        id = bbox['image_id']

        file_name = image_id_name_dict[id]
        image_name_bboxs_dict[file_name].append(bbox['bbox'])

    for image_name, bboxs in image_name_bboxs_dict.items():
        image_path = str(image_dir_path.joinpath(image_name).resolve())
        image = cv2.imread(image_path)
        image_drawed = draw_bbox_from_bbox_list(image, bboxs, color=color, thickness=thickness)

        output_img_path = str(output_dir.joinpath(image_name).resolve())
        cv2.imwrite(output_img_path, image_drawed)


def draw_bbox_from_whole_img_result_json(image_dir_path, output_dir, GT_json_file, result_json_file, color=(0, 0, 0), thickness=3):
    image_dir_path = Path(image_dir_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    images_info = GT_json_file['images']
    image_name_bboxs_dict = defaultdict(list)
    image_id_name_dict = {}

    for image_info in images_info:
        file_name = image_info['file_name']
        id = image_info['id']
        image_id_name_dict[id] = file_name

    for bbox_info in result_json_file:
        bbox = bbox_info['bbox']
        image_id = bbox_info['image_id']
        image_name = image_id_name_dict[image_id]

        image_name_bboxs_dict[image_name].append(bbox)

    for image_name, bboxs in image_name_bboxs_dict.items():
        image_path = str(image_dir_path.joinpath(image_name).resolve())
        image = cv2.imread(image_path)
        image_drawed = draw_bbox_from_bbox_list(image, bboxs, color=color, thickness=thickness)

        output_img_path = str(output_dir.joinpath(image_name).resolve())
        cv2.imwrite(output_img_path, image_drawed)


def draw_bbox_from_bbox_list(image, bbox_list, color=(0, 0, 0), thickness=3):
    image_ = image.copy()
    for bbox in bbox_list:
        x1, y1, w, h = bbox[:4]
        x1, y1, x2, y2 = int(x1), int(y1), int(x1 + w), int(y1 + h)
        cv2.rectangle(image_, (x1, y1), (x2, y2), color=color, thickness=thickness)

    return image_


##############
def non_max_suppression_fast(result_boxes, iou_threshold, perClass=True):
    '''
        boxes : coordinates of each box
        scores : score of each box
        iou_threshold : iou threshold(box with iou larger than threshold will be removed)
    '''

    boxes = result_boxes[:, :-2]
    classes = result_boxes[:, -2]
    scores = result_boxes[:, -1]

    if boxes.shape[0] == 0:
        return []

    # Init the picked box info
    pick = []

    # Box coordinate consist of left top and right bottom
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    # Compute area of each boxes
    area = w * h

    # Greedily select the order of box to compare iou
    idxs = np.argsort(scores)

    while len(idxs) > 0:
        last = len(idxs) - 1  # Scrore가 가장 높은 박스
        i = idxs[last]  # Score가 가장 높은 박스의 인덱스
        c = classes[i]
        pick.append(i)  # Score가 가장 높은 박스의 인덱스를 pick에 저장

        # With vector implementation, we can calculate fast
        xx1 = np.maximum(x1[i], x1[idxs[:last]])  # Score가 가장 높은 박스와 나머지 박스의 좌표 비교
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x1[i] + w[i], x1[idxs[:last]] + w[idxs[:last]])
        yy2 = np.minimum(y1[i] + h[i], y1[idxs[:last]] + h[idxs[:last]])

        w_ = np.maximum(0, xx2 - xx1 + 1)
        h_ = np.maximum(0, yy2 - yy1 + 1)
        intersection = w_ * h_

        # Calculate the iou
        iou = intersection / (area[idxs[:last]] + area[idxs[last]] - intersection)

        if perClass:  # perClass == true인 경우에는, class index가 같은 경우에만 삭제 대상
            outCheck = (iou > iou_threshold)
            sameClassCheck = (classes[idxs[:last]] == c)
            allCheck = sameClassCheck & outCheck
            idxs = np.delete(idxs, np.concatenate(([last], np.where(allCheck)[0])))
        else:
            idxs = np.delete(idxs, np.concatenate(([last], np.where(iou > iou_threshold)[0])))

    return result_boxes[pick]


def calcul_IOU(box, GT_box):
    x1 = max(box[0], GT_box[0])
    y1 = max(box[1], GT_box[1])
    x2 = min(box[2], GT_box[2])
    y2 = min(box[3], GT_box[3])

    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)
    intersection = w * h

    area1 = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area2 = (GT_box[2] - GT_box[0] + 1) * (GT_box[3] - GT_box[1] + 1)

    return intersection / (area1 + area2 - intersection)


def compare_gt_and_result(gt_boxes, result_boxes):
    gt_to_result_index_match_dict = {}
    result_to_gt_index_match_dict = {}

    # r_w = result_boxes[:, 2]
    # r_h = result_boxes[:, 3]
    # result_boxes_area = r_w * r_h
    # result_boxes_class = result_boxes[:, -2]
    #
    # for gt_index in range(gt_boxes.shape[0]):
    #     gt_box, gt_box_class = gt_boxes[gt_index, :-1], gt_boxes[gt_index, -1]
    #
    #     # gt와 같은 class의 result box가 없으면 바로 스킵
    #     same_class_box_index = (result_boxes_class == gt_box_class)
    #     if np.any(same_class_box_index) == False:
    #         continue
    #
    #     g_w = gt_box[2]
    #     g_h = gt_box[3]
    #     gt_box_area = g_w * g_h
    #
    #     same_class_result_box = result_boxes[same_class_box_index]
    #
    #     intersection_x1 = np.maximum(gt_box[0], result_boxes[:, 0])
    #     intersection_y1 = np.maximum(gt_box[1], result_boxes[:, 1])
    #     intersection_x2 = np.minimum(gt_box[0] + gt_box[2], result_boxes[:, 0] + result_boxes[:, 2])
    #     intersection_y2 = np.minimum(gt_box[1] + gt_box[3], result_boxes[:, 1] + result_boxes[:, 3])
    #
    #     intersection_w = np.maximum(0, intersection_x2 - intersection_x1 + 1)
    #     intersection_h = np.maximum(0, intersection_y2 - intersection_y1 + 1)
    #     intersection = intersection_w * intersection_h
    #
    #     iou = intersection / (gt_box_area + result_boxes_area - intersection)
    #
    #     over_IOU_index = (iou > 0.5)
    #
    #     over_iou_same_class = np.where(over_IOU_index & same_class_box_index)[0]
    #     if len(over_iou_same_class) != 0:
    #         max_score = -np.inf
    #         max_score_box_index = None
    #         for result_index in over_iou_same_class:
    #             result_box_score = result_boxes[result_index, -1]
    #             if max_score < result_box_score and (result_index not in result_to_gt_index_match_dict.keys()):
    #                 max_score = result_box_score
    #                 max_score_box_index = result_index
    #                 gt_to_result_index_match_dict[gt_index] = result_index
    #         result_to_gt_index_match_dict[max_score_box_index] = gt_index
    #     else:
    #         continue

    g_w = gt_boxes[:, 2]
    g_h = gt_boxes[:, 3]
    gt_boxes_area = g_w * g_h
    gt_boxes_class = gt_boxes[:, -1]

    result_boxes_score = result_boxes[:, -1]
    result_boxes_score_sorted_index = (-result_boxes_score).argsort()
    result_boxes_score_sorted = result_boxes[result_boxes_score_sorted_index]

    for result_index in range(result_boxes_score_sorted.shape[0]):
        result_box, result_box_class = result_boxes_score_sorted[result_index, :-2], result_boxes_score_sorted[result_index, -2]

        same_class_gt_box_index = (result_box_class == gt_boxes_class)
        if np.any(same_class_gt_box_index) == False:
            continue

        r_w = result_box[2]
        r_h = result_box[3]
        result_box_area = r_w * r_h

        intersection_x1 = np.maximum(result_box[0], gt_boxes[:, 0])
        intersection_y1 = np.maximum(result_box[1], gt_boxes[:, 1])
        intersection_x2 = np.minimum(result_box[0] + result_box[2], gt_boxes[:, 0] + gt_boxes[:, 2])
        intersection_y2 = np.minimum(result_box[1] + result_box[3], gt_boxes[:, 1] + gt_boxes[:, 3])

        intersection_w = np.maximum(0, intersection_x2 - intersection_x1 + 1)
        intersection_h = np.maximum(0, intersection_y2 - intersection_y1 + 1)
        intersection = intersection_w * intersection_h

        iou = intersection / (gt_boxes_area + result_box_area - intersection)

        iou_threshold = 0.5
        over_IOU_index = iou > iou_threshold
        iou_sorted_index = (-iou).argsort()

        iou_sorted_over_threshold_same_class_index = (over_IOU_index & same_class_gt_box_index)[iou_sorted_index]
        iou_sorted_same_class_over_iou_threshold_indexs = np.where(iou_sorted_over_threshold_same_class_index)[0]

        for iou_sorted_same_class_over_iou_threshold_index in iou_sorted_same_class_over_iou_threshold_indexs:
            real_gt_index = iou_sorted_index[iou_sorted_same_class_over_iou_threshold_index]
            real_result_index = result_boxes_score_sorted_index[result_index]

            if real_gt_index not in gt_to_result_index_match_dict.keys():
                gt_to_result_index_match_dict[real_gt_index] = real_result_index
                result_to_gt_index_match_dict[result_index] = real_gt_index
                break

    return gt_to_result_index_match_dict, result_to_gt_index_match_dict


def get_images_info_from_whole_GT_json(json_file):
    image_dict_list = json_file['images']
    image_id_to_name_dict = {}

    for image_dict in image_dict_list:
        file_name = image_dict['file_name']
        file_name_without_extension = file_name.split('.')[0]
        id = image_dict['id']

        image_id_to_name_dict[id] = file_name_without_extension

    return image_id_to_name_dict
##############


def draw_GT_Result():
    with open('./dataset_0_test_whole_image.json') as f:
        whole_img_dataset = json.load(f)

    with open('./rivision_experiment_1_result_whole_image.bbox.json') as f:
        whole_img_result = json.load(f)

    draw_bbox_from_whole_img_gt_json('../0. raw_data/original/EWP_Test/image', 'GT', whole_img_dataset)

    draw_bbox_from_whole_img_result_json('../0. raw_data/original/EWP_Test/image', 'Result', whole_img_dataset, whole_img_result)


def process_whole_image_gt_json(whole_img_GT_json):
    image_name_to_GT_bboxs_dict = defaultdict(list)

    image_id_to_name_dict = get_images_info_from_whole_GT_json(whole_img_GT_json)
    Gt_bboxs = whole_img_GT_json['annotations']
    for gt_bbox in Gt_bboxs:
        category_id = gt_bbox['category_id']
        image_id = gt_bbox['image_id']
        bbox = gt_bbox['bbox']

        image_name = image_id_to_name_dict[image_id]
        image_name_to_GT_bboxs_dict[image_name].append((bbox + [category_id]))

    image_name_to_GT_bboxs_dict = dict(image_name_to_GT_bboxs_dict)
    for image_name in image_name_to_GT_bboxs_dict.keys():
        image_name_to_GT_bboxs_dict[image_name] = np.asarray(image_name_to_GT_bboxs_dict[image_name])

    return image_name_to_GT_bboxs_dict


def process_whole_image_result_json(whole_img_GT_json, whole_img_result_json):
    image_name_to_Result_bboxs_dict = defaultdict(list)

    image_id_to_name_dict = get_images_info_from_whole_GT_json(whole_img_GT_json)
    for result_bbox in whole_img_result_json:
        category_id = result_bbox['category_id']
        score = result_bbox['score']
        image_id = result_bbox['image_id']
        bbox = result_bbox['bbox']

        image_name = image_id_to_name_dict[image_id]
        image_name_to_Result_bboxs_dict[image_name].append(bbox + [category_id, score])

    image_name_to_Result_bboxs_dict = dict(image_name_to_Result_bboxs_dict)
    for image_name in image_name_to_Result_bboxs_dict.keys():
        image_name_to_Result_bboxs_dict[image_name] = np.asarray(image_name_to_Result_bboxs_dict[image_name])

    return image_name_to_Result_bboxs_dict


def compare_GT_result():
    # GT 읽어들이기
    with open('./dataset_0_test_whole_image.json') as f:
        whole_img_GT_json = json.load(f)
    image_id_to_GT_bboxs_dict = process_whole_image_gt_json(whole_img_GT_json)

    # result json 읽어들이기
    with open('./rivision_experiment_1_result_whole_image.bbox.json') as f:
        whole_img_result_json = json.load(f)
    image_id_to_Result_bboxs_dict = process_whole_image_result_json(whole_img_GT_json, whole_img_result_json)

    # image_id_to_GT_bboxs_dict
    # image_id_to_Result_bboxs_dict
    # bboxs는 (num_of_bboxs, 5(x1, y1, w, h, class))

    print('image_id_to_GT_bboxs_dict')
    for k, v in image_id_to_GT_bboxs_dict.items():
        print(k, v.shape)

    print('\nimage_id_to_Result_bboxs_dict')
    for k, v in image_id_to_Result_bboxs_dict.items():
        after_nms_v = non_max_suppression_fast(v, 0.5)
        print(k, ', score > 0.5 : ', v.shape, ', after_nms_v : ', after_nms_v.shape)

        # path = '../0. raw_data/original/EWP_Test/image/{}.jpg'.format(k)
        # img = cv2.imread(path)
        # over_score_img = draw_bbox_from_bbox_list(img, v, color=(255, 0, 0), thickness=3)
        # after_nms_img = draw_bbox_from_bbox_list(img, after_nms_v, color=(0, 255, 0), thickness=3)
        #
        # cv2.imwrite('./Result_over_score_nms/{}_1_over_score.jpg'.format(k), over_score_img)
        # cv2.imwrite('./Result_over_score_nms/{}_2_after_nms_img.jpg'.format(k), after_nms_img)

    for imamge_name in image_id_to_GT_bboxs_dict.keys():
        gt_bboxes = image_id_to_GT_bboxs_dict[imamge_name]
        result_bboxes = image_id_to_Result_bboxs_dict[imamge_name]
        after_nms_result_bboxes = non_max_suppression_fast(result_bboxes, 0.5)

        gt_to_result_index_match_dict, result_to_gt_index_match_dict = compare_gt_and_result(gt_bboxes, after_nms_result_bboxes)

        path = '../0. raw_data/original/EWP_Test/image/{}.jpg'.format(imamge_name)
        ii = 0
        print(gt_to_result_index_match_dict)
        print(result_to_gt_index_match_dict)
        for gt_idx, result_idx in gt_to_result_index_match_dict.items():
            print(gt_idx, result_idx)
            img = cv2.imread(path)

            gt_box = gt_bboxes[gt_idx]
            result_box = after_nms_result_bboxes[result_idx]
            print(gt_box)
            print([int(i) for i in result_box])

            x1, y1, x2, y2 = gt_box[0], gt_box[1], gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=5)

            x1, y1, x2, y2 = int(result_box[0]), int(result_box[1]), int(result_box[0] + result_box[2]), int(result_box[1] + result_box[3])
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=3)
            print()
            cv2.imwrite('./GT_and_result/{}_{}.jpg'.format(imamge_name, ii), img)
            ii += 1



if __name__ == '__main__':
    compare_GT_result()


