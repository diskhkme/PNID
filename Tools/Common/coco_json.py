import json
from collections import defaultdict
from itertools import islice
from copy import deepcopy
from pathlib import Path


def coco_json_write(outpath, coco_data):
    with open(outpath, "w") as json_out:
        json.dump(coco_data, json_out, indent=4)

class coco_json_reader():
    """ 도면 인식 관련 json 기본 클래스

    Arguments:
        filepath (string): json 파일 경로

    """
    def __init__(self, filepath):
        self.filepath = filepath
        with open(filepath) as f:
            self.gt_json_data = json.load(f)

class coco_gt_json_reader(coco_json_reader):
    """ 도면 ground truth json 클래스 (test.json 파일 파싱)

    Arguments:
        filepath (string): test.json 파일 경로

    """
    def __init__(self, filepath):
        super().__init__(filepath)
        self.img_id_to_filename_dict = self.get_img_dict() # {1: ["KNU-A-22300-001-04", 0, 0], 2: ["KNU-A-22300-001-04, 0, 1], ...}

    def get_img_dict(self):
        image_dict_list = self.gt_json_data['images']
        img_id_to_filename_dict = {}

        for image_dict in image_dict_list:
            file_name = image_dict['file_name']
            file_name_without_extension = file_name.split('.')[0]
            true_filename, h, w = file_name_without_extension.split('_')
            id = image_dict['id']

            img_id_to_filename_dict[id] = [true_filename, int(h), int(w)]

        return img_id_to_filename_dict


class coco_dt_json_reader(coco_gt_json_reader):
    """ 도면 test 결과 json 클래스. 도면 분할 테스트 결과 병합을 위해 gt와 dt json를 모두 필요로 함

    Arguments:
        gt_filepath (string): test.json 파일 경로
        dt_filepath (string): mmdetection의 테스트 결과로 출력되는 json 파일 경로
        stride_w, stride_h (int): 학습 데이터 생성 당시 이미지 분할에 사용한 stride 파라메터터
        # TODO : 학습 데이터 생성 당시에 json의 "info" tag에 분할 stride를 적어놓으면 별도 입력이 필요 없을듯
    """
    def __init__(self, gt_filepath, dt_filepath, drawing_resize_scale, stride_w, stride_h):
        super().__init__(gt_filepath)
        with open(dt_filepath) as f:
            self.dt_json_data = json.load(f)

        self.drawing_resize_scale = drawing_resize_scale
        self.stride_w = stride_w
        self.stride_h = stride_h

        # img_id / filename / bbox / global bbox로 용어 통일
        self.img_id_to_bbox_dict = self.get_img_id_to_bbox_dict() # {(segmented)img_id : [(local)bbox], score, category_id}
        self.img_id_to_global_bbox_dict = self.convert_bbox_coordinate_to_global(self.drawing_resize_scale) # {(segmented)img_id : [(global)bbox], score, category_id}
        # 최종 결과 데이터. 도면 이름을 key로, 변환된 박스 정보들을 value로 가지고 있는 dict # TODO: 변환 기능을 별도 모듈로 분리
        self.filename_to_global_bbox_dict = self.get_filename_to_global_bbox_dict() # {(source)img_name : list([(global)bbox], score, category_id})}

    def get_img_id_to_bbox_dict(self):
        img_id_to_bbox_dict = defaultdict(list)

        for result_dict in self.dt_json_data:
            image_id = result_dict['image_id']
            img_id_to_bbox_dict[image_id].append(dict(islice(result_dict.items(), 1, None)))

        return img_id_to_bbox_dict

    def convert_bbox_coordinate_to_global(self, resize_scale):
        """ 분할 도면의 박스 좌표를 global 좌표(원본 도면 기준의 좌표)로 변환

        Arguments:
            resize_scale (float): 분할시에 사용한 resize scaling factor

        Return:
            도면의 id를 key로, 변환된 box 정보를 value로 갖는 dict
        """
        img_id_to_global_bbox_dict = deepcopy(self.img_id_to_bbox_dict)
        for image_id in img_id_to_global_bbox_dict.keys():
            bboxs_info = img_id_to_global_bbox_dict[image_id]
            image_name, h, w = self.img_id_to_filename_dict[image_id]

            for bbox_info in bboxs_info:
                bbox_in_grid = bbox_info['bbox']
                bbox_in_grid[0] = int((bbox_in_grid[0] + self.stride_w * w)/resize_scale)
                bbox_in_grid[1] = int((bbox_in_grid[1] + self.stride_h * h)/resize_scale)
                bbox_in_grid[2] = int(bbox_in_grid[2]/resize_scale)
                bbox_in_grid[3] = int(bbox_in_grid[3]/resize_scale)

        return img_id_to_global_bbox_dict

    def get_filename_to_global_bbox_dict(self):
        """ 편의성을 위해 key를 id에서 도면 이름으로 변환하여 반환

        Return:
            도면의 이름을 key로, 변환된 box 정보를 value로 갖는 dict
        """
        filename_to_global_bbox_dict = defaultdict(list)

        for image_id in self.img_id_to_global_bbox_dict.keys():
            bboxs_info = self.img_id_to_global_bbox_dict[image_id]
            image_name, h, w = self.img_id_to_filename_dict[image_id]

            filename_to_global_bbox_dict[image_name] += bboxs_info

        return filename_to_global_bbox_dict

if __name__ == '__main__':
    gt_filepath = "D:/Test_Models/PNID/EWP_Data/Wonyong_Segment/dataset_2/test.json"
    dt_filepath = "D:/Libs/Pytorch/SwinTransformer/workdir/wonyong/dataset_2/dataset_2_sgd.bbox.json"

    test_dt_json = coco_dt_json_reader(gt_filepath, dt_filepath, 300, 300)
    test_dt_json.write_global_bbox_json("global_bbox.json")