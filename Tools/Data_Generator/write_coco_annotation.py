import json

def write_coco_annotation(out_path, annotation_data, symbol_dict, segment_params):
    """ 저장된 분할 도면 정보를 기반으로 COCO 포맷의 json 파일 생성 # TODO: coco_json으로 이동 검토

    Arguments:
        out_path (string): 출력 파일명
        annotation_data (list): 출력 annotation 데이터
        symbol_dict (dict): 심볼 dictionaty (category id 매칭에 사용)
        segment_params (list): segmentation parameter ([width,height, width_stride, height_stride]
    """
    data = {}
    data["type"] = "instances"
    image_dict = construct_image_dict(annotation_data)

    data["images"] = []
    for img_name, idx in image_dict.items():
        data["images"].append({"file_name" : img_name, "width" : segment_params[0], "height" : segment_params[1], "id" : idx})

    data["annotations"] = []
    instance_id = 0
    for annotation in annotation_data: # [drawingname, symname, minx, miny, maxx, maxy]
        if annotation[1] < 0: # test/val set을 위한 예외 case
            continue

        area = (annotation[4]-annotation[2]) * (annotation[5]-annotation[3])

        bbox = [annotation[2],annotation[3],annotation[4]-annotation[2],annotation[5]-annotation[3]] # [x y width height]

        if annotation[1] in symbol_dict.values():
            data["annotations"].append({"id" : instance_id,
                                        "bbox" : bbox ,
                                       "image_id" : image_dict[annotation[0]],
                                       "segmentation" : [],
                                       "ignore" : 0,
                                       "area" : area,
                                       "iscrowd" : 0,
                                       "category_id": annotation[1]})
            instance_id = instance_id+1

    data["categories"] = []
    for key, val in symbol_dict.items():
        data["categories"].append({"id" : val,
                                   "name" : key})

    # data["annotations"].append({"bbox": [annotation[2], annotation[4], annotation[3], annotation[5]]})

    with open(out_path, "w") as json_out:
        json.dump(data, json_out, indent=4)

    return image_dict


def construct_image_dict(annotation_data):
    """
    Arguments:
        annotation_data (list): 전체 심볼 annotation 정보 [sub_img_name, symbol_name, xmin, ymin, xmax, ymax]
    :return:
        이미지에 대한 id dictionary ["이미지 이름" : id]
    """
    image_dict = {}
    image_id = 0
    for data in annotation_data:
        if data[0] in image_dict: # 키가 이미 존재하는 경우
            continue
        else:
            image_dict[data[0]] = image_id
            image_id = image_id + 1

    return image_dict