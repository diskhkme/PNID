import os, sys
import io
import numpy as np
from copy import deepcopy

from Common.coco_json import coco_dt_json_reader, coco_json_write
from Common.pnid_xml import symbol_xml_reader
from Common.symbol_io import read_symbol_txt

class gt_dt_data():
    """ Evaluate 및 성능 계산을 위한 모든 데이터를 저장하는 클래스

    Arguments:
        gt_json_filepath (string): ground truth json file 경로 (학습 데이터 분할 과정에서 도출된 test.json)
        dt_json_filepath (string): detection result json file 경로 (mmdetection을 통한 테스트 결과 json, 분할 도면)
        drawing_dir (string): 원본 도면 이미지 있는 폴더
        xml_dir (string): 원본 도면과 함께 제공된 xml파일이 있는 폴더
        symbol_filepath (string): 심볼 리스트 텍스트파일 (.txt)
        drawing_resize_scale (float): 도면 분할 과정에서 적용된 resize scale (가로세로 절반이면 0.5)
        stride_w, stride_h (int): 도면 분할 과정에서 사용한 width 및 height stride
        score_threshold (float): 테스트 결과에서, score < score_threshold이면 score_filter 과정에서 제거
        nms_threshold (float): NMS threshold
    """
    def __init__(self, gt_json_filepath, dt_json_filepath, drawing_dir, xml_dir, symbol_filepath,
                 drawing_resize_scale, stride_w, stride_h,
                 score_threshold = 0.5, nms_iou_threshold = 0.1):
        self.drawing_dir = drawing_dir
        self.xml_dir = xml_dir
        self.drawing_resize_scale = drawing_resize_scale

        self.symbol_dict = read_symbol_txt(symbol_filepath)
        self.nms_iou_threshold = nms_iou_threshold

        # 주요 생성 데이터-----------
        # 모든 데이터는 도면이름을 Key로, box 정보들을 value로 가지고 있음
        # 1) dt_result_raw: 분할 도면의 dt 결과를 좌표 변환만 한 데이터
        self.dt_result_raw = coco_dt_json_reader(gt_json_filepath, dt_json_filepath, drawing_resize_scale, stride_w, stride_h).filename_to_global_bbox_dict
        # 2) dt_result: 좌표 변환후 score 기반으로 필터링한 데이터
        self.dt_result = self.score_filter(score_threshold)
        # 3) gt_result: ground truth xml 데이터. (gt_result_json은 동일한 데이터를 coco json 형태로 변환한 것)
        self.gt_result_json, self.gt_result = self.parse_test_gt_xmls()
        # 4) dt_result_after_nms: score 기반으로 필터링 후 NMS를 수행한 데이터
        self.dt_result_after_nms = self.get_dt_result_nms(self.nms_iou_threshold)

    def score_filter(self, score_threshold):
        dt_result = {}
        for key, values in self.dt_result_raw.items():
            filtered_values = [x for x in values if x["score"] >= score_threshold]
            dt_result[key] = filtered_values

        return dt_result

    def parse_test_gt_xmls(self):
        """ Grount Truth를 저장하고 있는 xml들을 읽어서 coco json 형식과 dict 형식으로 반환

        Return:
             gt_json (dict): 모든 test 도면의 GT 정보가 coco 형식으로 저장된 dict
             gt_result (dict): 모든 test 도면의 GT 정보가 image_name을 key로, bbox들이 value로 저장된 dict
        """
        gt_json = {}
        gt_result_json = {}

        test_image_filenames = self.dt_result.keys()

        images = []
        annotations = []
        categories = []

        for sym_name, sym_id in self.symbol_dict.items():
            categories.append({"id" : sym_id,
                               "name" : sym_name})

        image_id = 1
        object_id = 1

        for test_image_filename in test_image_filenames:
            symbol_xml = symbol_xml_reader(os.path.join(self.xml_dir,f"{test_image_filename}.xml"))
            filename, width, height, depth, object_list = symbol_xml.getInfo()
            image = {
                "file_name": f"{test_image_filename}.jpg",
                "id": image_id,
                "width": width,
                "height": height # url, date 등 사용하지 않는 데이터 삭제
            }
            images.append(image)
            annotations_per_image = []
            for object in object_list: # [name, xmin, ymin, xmax, ymax]
                x = object[1]
                y = object[2]
                width = object[3] - object[1]
                height = object[4] - object[2]

                name = object[0].split(sep='-', maxsplit=1)[0]
                class_index = self.symbol_dict[name]

                obj = {
                    "bbox": [x,y, width, height],
                    "category_id": class_index,
                    "image_id": image_id, 'id': object_id, 'area': width * height, 'segmentation': [],
                    "iscrowd": 0,
                    "ignore": 0,
                }
                annotations.append(obj)
                annotations_per_image.append(obj)
                object_id += 1

            gt_result_json[test_image_filename] = annotations_per_image
            image_id += 1

        gt_json['annotations'] = annotations
        gt_json['images'] = images
        gt_json['categories'] = categories

        return gt_json, gt_result_json

    def get_dt_result_nms(self, nms_threshold):
        """ 모든 dt_result의 도면에 대해 nms 수행
        Arguments:
            nms_threshold (float): NMS threshild
        Return:
            각 filename을 key로 NMS 수행후의 bbox를 value로 가지고있는 dict
        """

        filename_to_global_bbox_dict_after_nms = {}
        for img, bbox in self.dt_result.items():

            nms_result = self.non_max_suppression_fast(bbox, nms_threshold)
            filename_to_global_bbox_dict_after_nms[img] = nms_result

        return filename_to_global_bbox_dict_after_nms

    # TODO : mmcv의 SoftNMX 사용
    def non_max_suppression_fast(self, result_boxes, iou_threshold, perClass=True):
        """ 도면별 Box list에 대해 NMS 수행

        Arguments:
            dict result_boxes: [bbox], category_id, image_id 등등을 모두 가지고있는 bbox dict
            float iou_threshold: NMS threshold
            bool perClass: true면 동일 클래스끼리만 NMS 수행, false면 클래스 상관없이 NMS 수행

        Return:
            NMS 후 남아있는 result_boxes의 부분집합 dict
        """

        boxes = np.array([x["bbox"] for x in result_boxes])
        classes = np.array([x["category_id"] for x in result_boxes])
        scores = np.array([x["score"] for x in result_boxes])

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

        return [result_boxes[i] for i in pick]

# if __name__ == '__main__':
#     gt_json_filepath = "D:/Test_Models/PNID/EWP_Data/Wonyong_Segment/dataset_2/test.json"
#     dt_json_filepath = "D:/Libs/Pytorch/SwinTransformer/workdir/wonyong/dataset_2/dataset_2_sgd.bbox.json"
#     drawing_dir = "D:/Test_Models/PNID/EWP_Data/Drawing"
#     xml_dir = "D:/Test_Models/PNID/EWP_Data/SymbolXML"
#     symbol_filepath = "D:/Test_Models/PNID/EWP_Data/Symbol Class List.pbtxt"
#     stride_w = 300
#     stride_h = 300
#
#     eval = gt_dt_data(gt_json_filepath, dt_json_filepath, drawing_dir, xml_dir, symbol_filepath, stride_w, stride_h)
