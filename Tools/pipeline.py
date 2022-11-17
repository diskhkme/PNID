import copy
import os.path

import mmcv
import time
import sys
from os import path, makedirs
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector
from mmdet.models import build_detector
from Data_Generator.generate_segmented_data import segment_image
from Common.pnid_xml import write_symbol_result_to_xml, write_text_result_to_xml
from Predict_Postprocess.text_recognition.recognize_text import get_text_detection_result, recognize_text
from Common.symbol_io import read_symbol_txt, read_symbol_type_txt
from Predict_Postprocess.gt_dt_data import non_max_suppression_fast

INPUT_DIR = './input_files/'
OUTPUT_DIR = './detection_results/'

CONFIG = INPUT_DIR + 'model_config.py'
CHECKPOINT = INPUT_DIR + 'latest.pth'
SYM_PATH = INPUT_DIR + 'Hyundai_SymbolClass_Sym_Only.txt'
SYM_TYPE_PATH = INPUT_DIR + 'Hyundai_SymbolClass_Type.txt'

segment_params = [800, 800, 300, 300]
drawing_resize_scale = 0.5
score_threshold = 0.5
nms_threshold = 0.0
matching_iou_threshold = 0.5  # 매칭(정답) 처리할 IOU threshold
adaptive_thr_dict = {
    311: 0.02, 66: 0.06, 145: 0.4,
    131: 0.65, 431: 0.04, 109: 0.03,
    239: 0.3, 12: 0.0002, 499: 0.2
}

text_img_margin_ratio = 0.1

def check_dir(image_path):
    if not path.isdir(INPUT_DIR):
        print('Input 폴더를 찾을 수 없습니다. 실행 경로를 확인해 주세요.')
        exit()
    if not path.isdir(OUTPUT_DIR):
        try:
            makedirs(OUTPUT_DIR)
        except:
            print('Output 폴더 생성 실패')
            exit()

def load_model(config: str, checkpoint: str):
    device='cuda:0'

    config = mmcv.Config.fromfile(config)
    # config.model.pretrained = None

    model = build_detector(config.model)

    checkpoint = load_checkpoint(model, checkpoint, map_location=device)

    model.CLASSES = checkpoint['meta']['CLASSES']
    model.cfg = config

    model.to(device)
    model.eval()

    return model

def score_filtering(result, score_threshold):
    filtered = []
    for category in result:
        filtered.append([x for x in category if x[4] >= score_threshold])
    
    return filtered

def change_obj_format(obj: list):
    '''
           [x_min, y_min, x_max, y_max, score] 
        -> [x_min, y_min, width, height, score] 
    '''
    return [obj[0], obj[1], obj[2] - obj[0], obj[3] - obj[1], obj[4]]

def convert_bbox_to_global(result, cur_w, cur_h, segment_params, resize_scale):
    """ 하나의 분할 도면의 감지된 박스 좌표를 global 좌표(원본 도면 기준의 좌표)로 변환

    Arguments:
        result (list): 이미지 하나의 Detection 결과
        cur_w (int): 현재 이미지의 원본 도면 상에서의 열
        cur_h (int): 현재 이미지의 원본 도면 상에서의 행
        segment_params (list): 분할 파라미터
        resize_scale (float): 분할시에 사용한 resize scaling factor
    """
    width, height, stride_w, stride_h = segment_params
    global_list = []
    
    category_id = 0
    # category: 클래스
    for category in result:
        # 감지한 오브젝트
        for item in category:
            item = change_obj_format(item)
            global_bbox = []
            global_bbox.append(int((item[0] + stride_w * cur_w) / resize_scale))
            global_bbox.append(int((item[1] + stride_h * cur_h) / resize_scale))
            global_bbox.append(int(item[2]/resize_scale))
            global_bbox.append(int(item[3]/resize_scale))
            global_bbox.append(int(item[4]))
            global_bbox.append(category_id)
            global_list.append(global_bbox)
        category_id += 1

    return global_list

def get_dict_result(result: list):
    ''' 도면 하나의 결과를 dictionary로 리턴
    
    '''
    result_boxes = []
    for item in result:
        x = item[0]
        y = item[1]
        width = item[2]
        height = item[3]

        obj = {
            'bbox': [x, y, width, height],
            'category_id': item[5],
            'score': item[4],
            'area': width * height,
            'segmentation': [],
            'iscrowd': 0,
            'ignore': 0,
        }

        result_boxes.append(obj)

    return result_boxes

def detect_segmented_imgs(model, seg_imgs: list, score_threshold: float):
    result = []
    i = 0
    for image in seg_imgs:
        cur_w = image['w']
        cur_h = image['h']
        img = image['img']
        #* 1. 분할된 도면 하나에 대해 inference 수행
        res = inference_detector(model, img)
        #* 2. score filtering
        filtered_res = score_filtering(res, score_threshold)
        #* 3. 전역 좌표로 변환
        global_res = convert_bbox_to_global(filtered_res, cur_w, cur_h, segment_params, drawing_resize_scale)
        #* 4. dictionary로 포맷 변환
        dict_res = get_dict_result(global_res)
        #* 5. 결과에 추가
        result.extend(dict_res)

    dt_results = {}

    #* out은 원래 filename 자리
    dt_results['out'] = result
    
    return dt_results

def get_dt_result_nms(dt_result, nms_threshold):
    """ 모든 dt_result의 도면에 대해 nms 수행
    Arguments:
        nms_threshold (float): NMS threshild
    Return:
        각 filename을 key로 NMS 수행후의 bbox를 value로 가지고있는 dict
    """
    filename_to_global_bbox_dict_after_nms = {}
    for img, bbox in dt_result.items():
        nms_result = non_max_suppression_fast(bbox, nms_threshold, True, adaptive_thr_dict)
        filename_to_global_bbox_dict_after_nms[img] = nms_result

    return filename_to_global_bbox_dict_after_nms

if __name__=='__main__':
    if len(sys.argv) != 2:
        print('! 입력 도면 파일 이름을 인자로 입력해 주세요')
        exit()
    
    image_name = sys.argv[1]
    image_path = INPUT_DIR + image_name

    start = time.time()
    print('* Starting inference...')

    check_dir(image_path)

    #! 1. 이미지 분할
    seg_imgs = segment_image(image_path, segment_params, drawing_resize_scale)
    seg_elapsed = time.time()
    print(f'* 이미지 분할 소요 시간: {seg_elapsed - start}')

    #! 2. 학습한 모델 로드
    model = load_model(CONFIG, CHECKPOINT)
    load_elapsed = time.time()
    print(f'* 모델 Load 소요 시간: {load_elapsed - seg_elapsed}')

    #! 3. Detection 수행
    results = detect_segmented_imgs(model, seg_imgs, score_threshold)
    detect_elapsed = time.time()
    print(f'* Detection 소요 시간: {detect_elapsed - load_elapsed}')

    #! 4. NMS 수행
    nms_results = get_dt_result_nms(results, matching_iou_threshold)
    nms_elapsed = time.time()
    print(f'* NMS 소요 시간: {nms_elapsed - detect_elapsed}')

    #! 5. Symbol / Type 종류 dictionary로 열기
    symbol_dict = read_symbol_txt(SYM_PATH, include_text_as_class=True, include_text_orientation_as_class=False)
    symbol_type_dict = read_symbol_type_txt(SYM_TYPE_PATH)

    #! 6. Text recognition 수행
    dt_result_after_nms_text_only = get_text_detection_result(nms_results, symbol_dict)
    dt_result_text = recognize_text(image_path, dt_result_after_nms_text_only, text_img_margin_ratio,
                                               symbol_dict)
    print() #* Progress bar 개행 문제 때문에 추가

    text_recognition_elapsed = time.time()
    print(f'* Text recognition 소요 시간: {text_recognition_elapsed - nms_elapsed}')

    #! 7. 결과 XML 빌드
    symbol_xml_root = write_symbol_result_to_xml(OUTPUT_DIR, nms_results, symbol_dict, symbol_type_dict)
    text_xml_root = write_text_result_to_xml(OUTPUT_DIR, dt_result_text, symbol_dict)

    # ! 8. 결과 단일(Symbol+Text) XML 출력
    import xml
    output_xml_root = copy.deepcopy(text_xml_root)
    for obj in symbol_xml_root.findall('symbol_object'):
        output_xml_root.append(obj)

    out_path = os.path.join(OUTPUT_DIR, f'{os.path.splitext(image_name)[0]}_result.xml')
    xml.etree.ElementTree.ElementTree(output_xml_root).write(out_path)

    print(f'* 총 소요 시간: {time.time() - start}')
