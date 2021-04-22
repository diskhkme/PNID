# Common
공통 기능
- pnid_xml: pnid_XML의 입력 및 출력 클래스
- coco_json: COCO 포맷 json의 입력 및 출력 클래스
- symbol_io: symbol dictionary 관련 모듈

# Data_Generator
PNID 도면 및 XML로부터 mmdetection에 사용 가능한 학습 데이터 생성
- generate_training_data: 학습 데이터 생성 메인 모듈
- generate_segmented_data: xml 기반 분할 정보 생성 및 분할 도면 출력 기능
- write_coco_annotation: 분할 annotation으로 COCO 포맷 json 생성 기능

# Predict_Postprocess
Prediction 결과의 성능 계산 및 가시화 등 후처리
- predict_postprocess: 후처리 메인 모듈
- gt_dt_data: test 결과(분할된 이미지 기준)의 원본 이미지 기준 변환 및 NMS 기능
- evaluate: test 결과를 기반으로 한 precision, recall, AP 계산 기능

# TextXML_Error_Correct
Text 학습 데이터의 오류 제거
- text 학습 데이터의 공백 trim, multiline 분할, 좌우 공백 제거 기능

# Visualize
도면 이미지에 데이터 오버레이 및 출력
- symbol_visualize: 심볼 인식 결과 디버깅을 위한 오버레이 및 출력 기능
- text_xml_visualize: 텍스트 xml 정보 오버레이 및 출력 기능
