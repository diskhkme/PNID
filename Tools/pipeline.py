import mmcv
import mmdet
import cv2
import pprint
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector
from Data_Generator.generate_segmented_data import segment_image

CONFIG = 'C:\\Users\\DongwonJeong\\Desktop\\HyundaiPNID\\mmdetection-2.25.1\\Dataset1009_srcnn_text_rotated\\sparse_rcnn_r50_fpn_1x_coco.py'
CHECKPOINT = 'C:\\Users\\DongwonJeong\\Desktop\\HyundaiPNID\\mmdetection-2.25.1\\Dataset1009_srcnn_text_rotated\\latest.pth'
IMG_PATH = 'C:\\Users\\DongwonJeong\\Desktop\\HyundaiPNID\\Data\\Drawing\\JPG\\26071-200-M6-052-00001.jpg'
segment_params = [800, 800, 300, 300]
drawing_resize_scale = 0.5

def load_model(config: str, checkpoint: str):
    device='cuda:0'

    config = mmcv.Config.fromfile(config)
    config.model.pretrained = None

    model = build_detector(config.model)

    checkpoint = load_checkpoint(model, checkpoint, map_location=device)

    model.CLASSES = checkpoint['meta']['CLASSES']
    model.cfg = config

    model.to(device)

    return model

def score_filter(result, score_threshold):
    dt_result = {}
    for key, values in result:
        filtered_values = [x for x in values if x["score"] >= score_threshold]
        dt_result[key] = filtered_values

    return dt_result

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

def detect_segmented_imgs(model, seg_imgs: list):
    pp = pprint.PrettyPrinter(indent=4)
    results = []
    for img in seg_imgs:
        res = inference_detector(model, img)
        score_filter(res)

        results.append(res)
        break
        # show_result_pyplot(model, img, result, score_thr=0.5)

    return results

if __name__=='__main__':
    img = cv2.imread(IMG_PATH)
    img = cv2.resize(img, dsize=(0,0), fx=drawing_resize_scale, fy=drawing_resize_scale, interpolation=cv2.INTER_LINEAR)

    seg_imgs = segment_image(img, segment_params)

    model = load_model(CONFIG, CHECKPOINT)
    results = detect_segmented_imgs(model, seg_imgs)