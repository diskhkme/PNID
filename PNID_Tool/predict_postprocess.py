from pathlib import Path
import shutil
import sys
import os

import src.Common.config as config
from src.Visualize.test_result_visualize import draw_test_results_to_img
from src.Predict_Postprocess.gt_dt_data import gt_dt_data
from src.Predict_Postprocess.evaluate import evaluate
from src.Common.pnid_xml import write_symbol_result_to_xml, write_text_result_to_xml
from src.Predict_Postprocess.text_recognition.recognize_text import get_text_detection_result, recognize_text_using_tess
from src.Common.symbol_io import read_symbol_type_txt


# Test 결과의 성능 계산 및 이미지 출력 코드
def detection_result_post_process(cfg):
    path_info = cfg['path_info']
    options = cfg['options']
    metric_param = cfg['metric_param']

    # vertical_threshold = 2 # 세로 문자열로 판단하는 기준. 세로가 가로보다 vertical_threshold 배 이상 길면 세로로 판단
    text_img_margin_ratio = 0.1  # detection된 문자열에서 크기를 약간 키워서 text recognition을 수행할 경우. (ex, 0.1이면 box를 1.1배 키워서 인식)

    # 1) gt_dt_data 클래스 초기화를 통한 데이터 전처리
    #   : 분할 Ground Truth(gt) 및 detection(dt) result를 기반으로 분할 전 도면 좌표로 다시 맵핑하고, score filtering, NMS를 수행
    gt_dt_result = gt_dt_data(cfg)

    #  small symbol result json 에 big symbol model result 추가
    if path_info['test_big_sym_dt_path']:
        gt_dt_result.merge_big_sym_result(path_info['test_gt_path'], path_info['test_big_sym_dt_path'], 0.125)

    symbol_dict = gt_dt_result.symbol_dict  # or read_symbol_txt(symbol_filepath)
    symbol_type_dict = read_symbol_type_txt(path_info['symbol_type_dict_path'])

    # 2) evaluate 클래스 초기화 및 매칭 정보 생성
    #   : NMS 완료된 dt result와 gt result간의 매칭 dictionary 생성
    eval = evaluate(path_info['output_dir'])
    gt_to_dt_match_dict, dt_to_gt_match_dict = eval.compare_gt_and_dt(gt_dt_result.gt_result,
                                                                      gt_dt_result.dt_result_after_nms,
                                                                      metric_param['matching_iou_threshold'])

    # 3) precision-recall 성능 및 AP 성능 계산 및 Dump
    #   : 위에서 얻은 정보들을 바탕으로 성능 계산(주의! AP 계산에는 NMS 하기 전의 결과가 전달되어야 함 (gt_dt_result.dt_result))
    pr_result = eval.calculate_pr(gt_dt_result.gt_result, gt_dt_result.dt_result_after_nms, gt_to_dt_match_dict)
    ap_result_str = eval.calculate_ap(gt_dt_result.gt_result_json, gt_dt_result.dt_result)
    eval.dump_pr_and_ap_result(pr_result, ap_result_str, gt_dt_result.symbol_dict)

    # --- (include_text_as_class == True 인 경우) Text recognition 수행 (오래걸림)
    if options['include_text_as_class'] == True:
        dt_result_after_nms_text_only = get_text_detection_result(gt_dt_result.dt_result_after_nms, symbol_dict)
        dt_result_text = recognize_text_using_tess(path_info['drawing_img_dir'], dt_result_after_nms_text_only, text_img_margin_ratio,
                                                   symbol_dict)
        gt_dt_result.dt_result_text_recognition = dt_result_text

    # --- PNID XML 형식으로 파일 출력
    #   : (주로) dt_result_after_nms를 출력하며, 필요에 따라 다른 단계의 데이터도 PNID XML형식으로 출력 가능
    write_symbol_result_to_xml(path_info['output_dir'], gt_dt_result.dt_result_after_nms, symbol_dict, symbol_type_dict)
    if options['include_text_as_class'] == True:
        write_text_result_to_xml(path_info['output_dir'], gt_dt_result.dt_result_text_recognition, symbol_dict)

    # --- 가시적으로 확인하기 위한 이미지 도면 출력
    #   : 8번의 경우 텍스트 인식이 되지 않은 경우 출력 불가
    draw_test_results_to_img(gt_dt_result, gt_to_dt_match_dict, dt_to_gt_match_dict,
                             path_info['drawing_img_dir'], path_info['output_dir'], modes=[1, 2, 3, 4, 5, 6, 7, 8], thickness=5)

if __name__ == '__main__':
    cfg_path = 'configs/predict_postprocess/EWP/predict_postprocess.yaml'
    if len(sys.argv) > 1:
        cfg_path = sys.argv[1]

    cfg = config.get_predict_postprocess_config(cfg_path)

    # 사용한 parameter 확인을 위해 이 python script 같이 저장
    target_path_to_cp = os.path.join(cfg['path_info']['output_dir'], os.path.basename(cfg_path))
    shutil.copyfile(cfg_path, target_path_to_cp)

    detection_result_post_process(cfg)



