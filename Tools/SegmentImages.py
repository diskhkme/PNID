import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from XMLReader import XMLReader

def segment_images_in_dataset(xml_list, drawing_folder, drawing_segment_folder, segment_params):
    """
    :param xml_list: xml 파일 리스트
    :param drawing_folder: 원본 도면 이미지가 존재하는 폴더
    :param drawing_segment_folder: 분할된 이미지를 저장할 폴더
    :param segment_params: 분할 파라메터 [가로 크기, 세로 크기, 가로 stride, 세로 stride]
    :return:
        xml에 있는 전체 도면에서 분할된 도면의 전체 정보 [sub_img_name, symbol_name, xmin, ymin, xmax, ymax]
    """
    entire_segmented_info = []

    for xmlPath in xml_list:
        print(f"Proceccing {xmlPath}...")
        fname, ext = os.path.splitext(xmlPath)
        if ext.lower() != ".xml":
            continue

        xmlReader = XMLReader(xmlPath)
        img_filename, width, height, depth, objectList = xmlReader.getInfo()

        img_file_path = os.path.join(drawing_folder, img_filename)
        segmented_objects_info = segment_images(img_file_path, drawing_segment_folder, objectList, segment_params)
        entire_segmented_info.extend(segmented_objects_info)

    return entire_segmented_info

def segment_images(img_path, seg_out_path, objects, segment_params):
    """
    :param img_path: 원본 이미지 경로
    :param seg_out_path : 출력 이미지 경로
    :param objects: 원본 이미지의 object list [symbol_name, xmin, ymin, xmax, ymax]
    :param segment_params: 분할 파라메터 [가로 크기, 세로 크기, 가로 stride, 세로 stride]
    :return:
        seg_obj_info : 한 도면에서 분할된 도면의 전체 정보 [sub_img_name, symbol_name, xmin, ymin, xmax, ymax]
    """
    width_size = segment_params[0]
    height_size = segment_params[1]
    width_stride = segment_params[2]
    height_stride = segment_params[3]

    bbox_array = np.zeros((len(objects),4))
    for ind in range(len(objects)):
        bbox_object = objects[ind]
        bbox_array[ind, :] = np.array([bbox_object[1], bbox_object[2], bbox_object[3], bbox_object[4]])

    img = cv2.imread(img_path)
    height_step = img.shape[0] // height_stride
    width_step = img.shape[1] // width_stride

    seg_obj_info = []
    for h in range(height_step):
        for w in range(width_step):
            start_width = width_stride * w
            start_height = height_stride * h

            xmin_in = bbox_array[:, 0] > start_width
            ymin_in = bbox_array[:, 1] > start_height
            xmax_in = bbox_array[:, 2] < start_width+width_size
            ymax_in = bbox_array[:, 3] < start_height+height_size
            is_bbox_in = xmin_in & ymin_in & xmax_in & ymax_in
            in_bbox_ind = [i for i, val in enumerate(is_bbox_in) if val == True]
            if len(in_bbox_ind) == 0:
                continue

            if start_width+width_size > img.shape[1]:
                sub_img = np.zeros((height_size,width_size,3))
                sub_img[:, 0:img.shape[1]-(start_width+width_size), :] = img[start_height:start_height+height_size, start_width:img.shape[1],:]
            elif start_height+height_size > img.shape[0]:
                sub_img = np.zeros((height_size, width_size, 3))
                sub_img[0:img.shape[0] - (start_height + height_size), : , :] = img[start_height:img.shape[0], start_width:start_width+width_size, :]
            else:
                sub_img = img[start_height:start_height+height_size, start_width:start_width+width_size, :]

            filename, _ = os.path.splitext(os.path.basename(img_path))
            sub_img_filename = f"{filename}_{w}_{h}.jpg"
            cv2.imwrite(os.path.join(seg_out_path,sub_img_filename), sub_img)

            for i in in_bbox_ind:
                seg_obj_info.append([sub_img_filename, objects[i][0], objects[i][1] - start_width, objects[i][2] - start_height,
                                     objects[i][3] - start_width, objects[i][4] - start_height])

            # fig, ax = plt.subplots(1)
            # ax.imshow(sub_img)
            # for i in in_bbox_ind:
            #     symbolxmin = objects[i][2] - start_width
            #     symbolxmax = objects[i][4] - start_width
            #     symbolymin = objects[i][3] - start_height
            #     symbolymax = objects[i][5] - start_height
            #     rect = patches.Rectangle((symbolxmin, symbolymin),
            #                              symbolxmax - symbolxmin,
            #                              symbolymax - symbolymin,
            #                              linewidth=1, edgecolor='r', facecolor='none')
            #     ax.add_patch(rect)
            # plt.show()

    return seg_obj_info