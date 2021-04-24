import os, sys, io
import numpy as np
from collections import defaultdict
from pycocotools import coco, cocoeval

from Common.coco_json import coco_json_write

class evaluate():
    """ Precision-Recall, AP 성능 계산 및 결과 Dump

    Arguments:
        output_dir (string) : dump 및 중간 과정에서 생성되는 COCO 형식의 dt, gt 데이터가 저장될 폴더
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def compare_gt_and_dt(self, gt_result, dt_result, matching_iou_threshold):
        # TODO : 완전한 결과를 보려면 matching이 되지않은 dt의 정보도 전달되면 좋을듯
        """ GT와 DT 결과를 비교하여 매칭이 성공한 index들을 dict로 반환하는 함수

        Arguments:
            gt_result (dict): 도면 이름을 key로, box들을 value로 갖는 gt dict
            dt_result (dict): 도면 이름을 key로, box들을 value로 갖는 dt dict
            matching_iou_threshold (float): IOU > threshold이고 카테고리(class)가 같으면 매칭된 것으로 처리
        Returns:
            gt_to_dt_match_dict (dict): gt의 심볼 index를 key로, 매칭된 dt의 심볼 index를 value로 갖는 dict
            dt_to_gt_match_dict (dict): dt의 심볼 index를 key로, 매칭된 gt의 심볼 index를 value로 갖는 dict
        """
        gt_to_dt_match_dict = {}
        dt_to_gt_match_dict = {}

        for filename, annotations in dt_result.items():
            gt_to_result_index_match_dict = {}
            result_to_gt_index_match_dict = {}

            gt_boxes = [x["bbox"] for x in gt_result[filename]]
            gt_boxes = np.array(gt_boxes)
            g_w = gt_boxes[:, 2]
            g_h = gt_boxes[:, 3]
            gt_boxes_area = g_w * g_h
            gt_boxes_class = [x["category_id"] for x in gt_result[filename]]
            gt_boxes_class = np.array(gt_boxes_class)

            boxes = np.array([x["bbox"] for x in annotations])
            classes = np.array([x["category_id"] for x in annotations])
            scores = np.array([x["score"] for x in annotations])
            result_boxes = np.zeros((boxes.shape[0], boxes.shape[1] + 2))
            result_boxes[:,0:4] = boxes
            result_boxes[:,4] = classes
            result_boxes[:,5] = scores

            result_boxes_score = result_boxes[:, -1]
            result_boxes_score_sorted_index = (-result_boxes_score).argsort()
            result_boxes_score_sorted = result_boxes[result_boxes_score_sorted_index]

            for result_index in range(result_boxes_score_sorted.shape[0]):
                result_box, result_box_class = result_boxes_score_sorted[result_index, :-2], result_boxes_score_sorted[
                    result_index, -2]

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

                iou_threshold = matching_iou_threshold
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

            gt_to_dt_match_dict[filename] = gt_to_result_index_match_dict
            dt_to_gt_match_dict[filename] = result_to_gt_index_match_dict

        return gt_to_dt_match_dict, dt_to_gt_match_dict

    def dump_pr_and_ap_result(self, pr_result, ap_result_str, symbol_dict):
        """ AP와 PR 계산 결과를 파일로 출력. test 내에 존재하는 모든 도면에 대해 한 파일로 한꺼번에 출력함

        Arguments:
            pr_result (dict): 도면 이름을 key로, 각 도면에서의 PR 계산에 필요한 정보들(detected_num, gt_num 및 클래스별 gt/dt num)을 저장한 dict
            ap_result_str (string): cocoeval의 evaluate summary를 저장한 문자열
            symbol_dict (dict): 심볼 이름을 key로, id를 value로 갖는 dict
        Returns:
            None
        """
        outpath = os.path.join(self.output_dir, "test_result.txt")
        with open(outpath, 'w') as f:
            mean_precision = 0
            mean_recall = 0

            for filename, values in pr_result.items():
                f.write(f"test drawing : {filename}----------------------------------\n")
                f.write(f"precision : {values['detected_num']} / {values['all_prediction_num']} = {values['precision']}\n")
                f.write(f"recall : {values['detected_num']} / {values['all_gt_num']} = {values['recall']}\n")
                mean_precision += values['precision']
                mean_recall += values['recall']

                for gt_class, gt_num, detected_num in zip(values["gt_classes"], values["per_class_gt_num"],values["per_class_detected_num"]):
                    if symbol_dict is not None:
                        sym_name = [k for k,v in symbol_dict.items() if v == gt_class]
                    else:
                        sym_name = ""
                    f.write(f"class {gt_class} ({sym_name}) : {detected_num} / {gt_num}\n")

                f.write("\n")
            f.write(ap_result_str)

            mean_precision /= len(pr_result.keys())
            mean_recall /= len(pr_result.keys())

            ap_strs = ap_result_str.splitlines()[0].split(" ")
            ap = float(ap_strs[len(ap_strs)-1])
            ap_50_strs = ap_result_str.splitlines()[1].split(" ")
            ap_50 = float(ap_50_strs[len(ap_50_strs) - 1])
            ap_75_strs = ap_result_str.splitlines()[2].split(" ")
            ap_75 = float(ap_75_strs[len(ap_75_strs) - 1])

            f.write(f"(mean precision, mean recall, ap, ap50, ap75) = ({mean_precision}, {mean_recall}, {ap}, {ap_50}, {ap_75})")

    def calculate_ap(self, gt_result_json, dt_result):
        """ COCOeval을 사용한 AP계산. 중간 과정으로 gt와 dt에 대한 json파일이 out_dir에 생성됨

        Arguments:
            gt_result_json (dict): test 내에 존재하는 모든 도면에 대한 images, annotation, category 정보를 coco json 형태로 저장한 dict
            dt_result (dict): 도면 이름을 key로, box들을 value로 갖는 dt dict
        Returns:
            result_str (string): COCOeval의 계산 결과 summary 저장한 문자열
        """
        # 먼저 gt_json을 파일로 출력
        gt_outpath = os.path.join(self.output_dir, "test_gt_global.json")
        coco_json_write(gt_outpath, gt_result_json)

        # dt_result를 coco형식으로 변환하여 파일로 출력 (주의! dt는 NMS 이전의 결과여야 함)
        test_dt_global = []
        for filename, bboxes in dt_result.items():
            for box in bboxes:
                box["image_id"] = self.get_gt_img_id_from_filename(filename, gt_result_json)
                test_dt_global.append(box)

        dt_outpath = os.path.join(self.output_dir, "test_dt_global.json")
        coco_json_write(dt_outpath, test_dt_global)

        # gt와 dt파일을 로드하여 ap 계산
        cocoGT = coco.COCO(gt_outpath)
        cocoDt = cocoGT.loadRes(dt_outpath)
        annType = 'bbox'
        cocoEval = cocoeval.COCOeval(cocoGT,cocoDt,annType)
        cocoEval.evaluate()
        cocoEval.accumulate()

        original_stdout = sys.stdout
        string_stdout = io.StringIO()
        sys.stdout = string_stdout
        cocoEval.summarize()
        sys.stdout = original_stdout

        result_str = string_stdout.getvalue()

        return result_str

    def calculate_pr(self, gt_result, dt_result, gt_to_dt_match_dict):
        """ 전체 test 도면에 대한 precision 및 recall 계산

        Arguments:
            gt_result (dict): 도면 이름을 key로, box들을 value로 갖는 gt dict
            dt_result (dict): 도면 이름을 key로, box들을 value로 갖는 dt dict
            gt_to_dt_match_dict (dict): gt 심볼과 dt 심볼간 매칭 계산 결과 dict

        Returns:
            pr_result (dict): 도면 이름을 key로, Precision 및 recall 계산에 필요한 데이터를 value로 갖는 dict
        """
        pr_result = {}
        for filename, bboxes in dt_result.items():

            gt_class_num_dict = defaultdict(int)
            gt_detected_class_num_dict = defaultdict(int)

            detected_num = 0
            all_prediction = len(bboxes)

            for gt_annotation in gt_result[filename]:
                gt_class_num_dict[gt_annotation["category_id"]] += 1

            gt_num = len(gt_result[filename])

            for gt_index in gt_to_dt_match_dict[filename].keys():
                gt_bbox = gt_result[filename][gt_index]
                gt_class = gt_bbox["category_id"]
                gt_detected_class_num_dict[gt_class] += 1
                detected_num += 1

            gt_classes = sorted(list(gt_class_num_dict.keys()))

            per_class_gt_num = []
            per_class_detected_num = []
            for class_index in gt_classes:
                per_class_gt_num.append(gt_class_num_dict[class_index])
                per_class_detected_num.append(gt_detected_class_num_dict[class_index])

            pr_result[filename] = { "all_gt_num" : gt_num,
                                    "all_prediction_num" : all_prediction,
                                    "detected_num" : detected_num,
                                    "gt_classes" : gt_classes,
                                    "per_class_detected_num" : per_class_detected_num,
                                    "per_class_gt_num" : per_class_gt_num,
                                    "precision" : detected_num/all_prediction,
                                    "recall" : detected_num/gt_num
                                    }

        return pr_result

    def get_gt_img_id_from_filename(self, filename, gt_result_json):
        """ filename을 입력으로 img_id를 반환하는 함수

        Arguments:
            filename (string): 파일이름 (ex, "KNU-A-36420-014.jpg")
            gt_result_json (dict): test 내에 존재하는 모든 도면에 대한 images, annotation, category 정보를 coco json 형태로 저장한 dict
        Return:
            gt_result_json에 기록된 filename에 해당하는 도면의 id
        """
        for imgs in gt_result_json['images']:
            if filename == imgs["file_name"].split(".")[0]:
                return imgs["id"]