from Visualize.symbol_visualize import draw_test_results_to_img
from Predict_Postprocess.gt_dt_data import gt_dt_data
from Predict_Postprocess.evaluate import evaluate

# Test 결과의 성능 계산 및 이미지 출력 코드 (#TODO: PNID XML 포맷으로 결과 출력 기능 구현)

gt_json_filepath = "D:/Test_Models/PNID/EWP_Data/Wonyong_Segment/dataset_1/test.json" # 학습 도면 분할시 생성한 test.json 파일 경로
dt_json_filepath = "D:/Libs/Pytorch/SwinTransformer/workdir/wonyong/dataset_1/GFL/gfl_dataset_1_sgd.bbox.json" # prediction 결과로 mmdetection에서 생성된 json 파일 경로
drawing_dir = "D:/Test_Models/PNID/EWP_Data/Drawing" # 원본 도면 이미지 폴더
xml_dir = "D:/Test_Models/PNID/EWP_Data/SymbolXML" # 원본 도면 이미지와 함께 제공된 Symbol XML 폴더
symbol_filepath = "D:/Test_Models/PNID/EWP_Data/EWP_SymbolClass_sym_only.txt" # (방향 제거된) symbol index txt 파일 경로
output_dir = "./" # 출력 파일들이 저장될 폴더
stride_w = 300 # 학습 도면 분할시에 사용한 stride
stride_h = 300
drawing_resize_scale = 0.5 # 학습 도면 분할시에 사용한 scaling factor (절반 크기로 줄였으면 0.5)
score_threshold = 0.5 # score filtering threshold
nms_threshold = 0.0
matching_iou_threshold = 0.5 # 매칭(정답) 처리할 IOU threshold


# 1) gt_dt_data 클래스 초기화를 통한 데이터 전처리
#   : 분할 Ground Truth(gt) 및 detection(dt) result를 기반으로 분할 전 도면 좌표로 다시 맵핑하고, score filtering, NMS를 수행
gt_dt_result = gt_dt_data(gt_json_filepath, dt_json_filepath, drawing_dir, xml_dir, symbol_filepath,
                       drawing_resize_scale, stride_w, stride_h,
                       score_threshold, nms_threshold)

# 2) evaluate 클래스 초기화 및 매칭 정보 생성
#   : NMS 완료된 dt result와 gt result간의 매칭 dictionary 생성
eval = evaluate(output_dir)
gt_to_dt_match_dict, dt_to_gt_match_dict = eval.compare_gt_and_dt(gt_dt_result.gt_result, gt_dt_result.dt_result_after_nms, matching_iou_threshold)

# 3) precision-recall 성능 및 AP 성능 계산
#   : 위에서 얻은 정보들을 바탕으로 성능 계산(주의! AP 계산에는 NMS 하기 전의 결과가 전달되어야 함 (gt_dt_result.dt_result))
pr_result = eval.calculate_pr(gt_dt_result.gt_result,gt_dt_result.dt_result_after_nms, gt_to_dt_match_dict)
ap_result_str = eval.calculate_ap(gt_dt_result.gt_result_json, gt_dt_result.dt_result)
eval.dump_pr_and_ap_result(pr_result,ap_result_str, gt_dt_result.symbol_dict)

# 4) 가시적으로 확인하기 위한 이미지 도면 출력
draw_test_results_to_img(gt_dt_result, gt_to_dt_match_dict, dt_to_gt_match_dict,
                         drawing_dir, output_dir, modes=(1,2,3,4,5,6,7), thickness=5)