# Data generator

## 사용 방법

### 데이터 폴더 구조

아래와 같이 데이터가 저장되어 있다고 가정함.

    <base_dir>
    ├── Drawing                             # 원본 도면 이미지 저장 폴더
    │   ├── KNU-A-22300-001-01.jpg
    │   ├── ...
    ├── SymbolXML                           # 원본 도면 Symbol Annotation 저장 폴더
    │   ├── KNU-A-22300-001-01.xml
    │   ├── ...
    ├── TextXML                             # 원본 도면 Text Annotation 저장 폴더
    │   ├── KNU-A-22300-001-01.xml
    │   ├── ...
    └── EWP_SymbolClass_sym_only.txt        # 심볼 클래스 리스트 파일(txt)
    
### 코드 실행

generate_training_data.py 에서 필요한 파라메터 및 경로 지정 후 실행

- base_dir 경로 지정
- drawing_segment_dir 경로 지정
- test_drawing list 설정
- 텍스트 포함 출력 여부, train/val ratio, 분할 파라메터, drawing resize scale 설정
    
### 출력 폴더 구조

drawing_segment_dir에 아래와 같은 파일들이 생성

    <base_dir>
    ├── ...                                     # 원본 도면 이미지 저장 폴더
    ├── <drawing_segment_dir>                   # 학습용 분할데이터 저장 폴더
    │   ├── train
    │   │   ├── KNU-A-22300-001-01_5_11.jpg     # 학습용 분할 이미지 (train은 심볼이 없는 이미지 파일은 저장 X)
    │   │   ├── ... 
    │   ├── val
    │   │   ├── KNU-A-22300-001-02_0_0.jpg      # 검증용 분할 이미지
    │   │   ├── ... 
    │   ├── test
    │   │   ├── KNU-A-22300-001-02_0_0.jpg      # 테스트용 분할 이미지
    │   │   ├── ...
    │   ├── train.json                          # train annotation(COCO json)
    │   ├── val.json                            # val annotation(COCO json)
    └── └── test.json                           # test annotation(COCO json)


    