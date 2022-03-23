import os
import sys
import shutil

import src.Common.config as config
from src.Data_Generator.generate_segmented_data import generate_segmented_data
from src.Common.symbol_io import read_symbol_txt
from src.Data_Generator.write_coco_annotation import write_coco_annotation

# 학습 데이터 생성 코드. 도면을 train/test/val로 나누고, 각 set의 이미지를 분할하여 sub_img들로 만들어 저장함
# 이때, train의 경우 심볼(또는 옵션에 따라 심볼+텍스트)가 존재하지 않는 도면은 저장하지 않음
# 단 test/val 도면의 경우 심볼이 존재하지 않아도 저장함

def generate_phase_data(phase,xmls, args):
    print(f"Generating {phase}...")
    annotation_data = generate_segmented_data(phase,xmls,*args[0:8])
    write_coco_annotation(os.path.join(args[2], "{}.json".format(phase)), annotation_data, args[0],args[3])


def generate_train_dataset(cfg):
    cfg_path = cfg['path_info']
    cfg_drawing = cfg['val_test_ignore_def']
    cfg_options = cfg['options']

    if cfg_drawing['train_drawings'] == None:
        train_drawings = [x.split(".")[0] for x in os.listdir(cfg_path['symbol_xml_dir'])
                          if x.split(".")[0] not in cfg_drawing['test_drawings'] and
                          x.split(".")[0] not in cfg_drawing['val_drawings'] and
                          x.split(".")[0] not in cfg_drawing['ignore_drawing']]

    symbol_dict = read_symbol_txt(cfg_path['symbol_dict_path'], cfg_options['include_text_as_class'], cfg_options['include_text_orientation_as_class'])

    train_xmls = [os.path.join(cfg_path['symbol_xml_dir'], f"{x}.xml") for x in train_drawings]
    val_xmls = [os.path.join(cfg_path['symbol_xml_dir'], f"{x}.xml") for x in cfg_drawing['val_drawings']]
    test_xmls = [os.path.join(cfg_path['symbol_xml_dir'], f"{x}.xml") for x in cfg_drawing['test_drawings']]

    segment_params = [cfg_options['segment_width'], cfg_options['segment_height'], cfg_options['segment_stride_w'], cfg_options['segment_stride_h']]

    common_args = (symbol_dict,
          cfg_path['drawing_img_dir'],
          cfg_path['drawing_segment_img_out_dir'],
          segment_params,
          cfg_options['drawing_resize_scale'],
          cfg_options['include_text_as_class'],
          cfg_options['include_text_orientation_as_class'],
          cfg_path['text_xml_dir'],)

    generate_phase_data("val", val_xmls, common_args)
    generate_phase_data("train", train_xmls, common_args)
    generate_phase_data("test", test_xmls, common_args)


if __name__ == '__main__':
    cfg_path = 'configs/generate_training_data/EWP/S05_800_400.yaml'
    if len(sys.argv) > 1:
        cfg_path = sys.argv[1]

    cfg = config.get_generate_training_data_config(cfg_path)

    # copy cfg to output folder
    target_path_to_cp = os.path.join(cfg['path_info']['drawing_segment_img_out_dir'], os.path.basename(cfg_path))
    shutil.copyfile(cfg_path, target_path_to_cp)

    generate_train_dataset(cfg)








