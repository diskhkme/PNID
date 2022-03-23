import os
import os.path as osp
import yaml

def _check_dir(dir, make_dir=True):
    if not osp.exists(dir):
        if make_dir:
            print('Create directory {}'.format(dir))
            os.mkdir(dir)
        else:
            raise Exception('Directory not exist {}'.format(dir))

def get_generate_training_data_config(config_file):
    with open(config_file, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.loader.SafeLoader)

    # relative path to absolute
    cfg['path_info']['drawing_img_dir'] = os.path.join(cfg['path_info']['base_dir'],cfg['path_info']['drawing_img_dir'])
    cfg['path_info']['drawing_segment_img_out_dir'] = os.path.join(cfg['path_info']['base_dir'],cfg['path_info']['drawing_segment_img_out_dir'])
    cfg['path_info']['symbol_xml_dir'] = os.path.join(cfg['path_info']['base_dir'],cfg['path_info']['symbol_xml_dir'])
    cfg['path_info']['text_xml_dir'] = os.path.join(cfg['path_info']['base_dir'],cfg['path_info']['text_xml_dir'])
    cfg['path_info']['symbol_dict_path'] = os.path.join(cfg['path_info']['base_dir'],cfg['path_info']['symbol_dict_path'])

    _check_dir(cfg['path_info']['drawing_segment_img_out_dir'], make_dir=True)

    return cfg

# def get_train_config(config_file='config/train_config_res_gcn.yaml'):
#     with open(config_file, 'r') as f:
#         cfg = yaml.load(f, Loader=yaml.loader.SafeLoader)
#
#     _check_dir(cfg['dataset']['data_path'], make_dir=False)
#     _check_dir(cfg['ckpt_root'])
#
#     return cfg
#
# def get_test_config(config_file='config/test_config.yaml'):
#     with open(config_file, 'r') as f:
#         cfg = yaml.load(f, Loader=yaml.loader.SafeLoader)
#
#     return cfg