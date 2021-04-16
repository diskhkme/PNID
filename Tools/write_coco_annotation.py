import json

def write_coco_annotation(outfile, annotation_data, symbol_dict, segment_params):
    data = {}
    data["type"] = "instances"
    image_dict = construct_image_dict(annotation_data)

    data["images"] = []
    for img_name, idx in image_dict.items():
        data["images"].append({"file_name" : img_name, "width" : segment_params[0], "height" : segment_params[1], "id" : idx})

    data["annotations"] = []
    instance_id = 0
    for annotation in annotation_data:
        area = (annotation[4]-annotation[2]) * (annotation[5]-annotation[3])
        bbox = [annotation[2],annotation[4],annotation[3]-annotation[2],annotation[5]-annotation[4]] # [x y width height]

        data["annotations"].append({"id" : instance_id,
                                    "bbox" : bbox ,
                                   "image_id" : image_dict[annotation[0]],
                                   "segmentation" : [],
                                   "ignore" : 0,
                                   "area" : area,
                                   "iscrowd" : 0,
                                   "category_id": symbol_dict[annotation[1]]})
        instance_id = instance_id+1

    data["categories"] = []
    for key, val in symbol_dict.items():
        data["categories"].append({"id" : val,
                                   "name" : key})

        # data["annotations"].append({"bbox": [annotation[2], annotation[4], annotation[3], annotation[5]]})

    with open(outfile, "w") as json_out:
        json.dump(data, json_out, indent=4)


def construct_image_dict(annotation_data):
    """
    :param annotation_data: 전체 심볼 annotation 정보 [sub_img_name, symbol_name, xmin, ymin, xmax, ymax]
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