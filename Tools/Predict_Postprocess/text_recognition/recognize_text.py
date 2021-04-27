import os
import cv2
import pytesseract
from Common.print_progress import print_progress
from copy import deepcopy
import matplotlib.pyplot as plt # debug purpose

def get_text_detection_result(dt_result, symbol_dict):
    bboxes_text = {}
    for filename, bboxes in dt_result.items():
        bboxes_text[filename] = [x for x in bboxes if x["category_id"] == symbol_dict["text"] or x["category_id"] == symbol_dict["text_rotated"]]

    return bboxes_text

def recognize_text_using_tess(drawing_dir, dt_result_after_nms_text_only, text_img_margin_ratio, symbol_dict):
    pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract'
    dt_result_text = deepcopy(dt_result_after_nms_text_only)
    for filename, bboxes in dt_result_text.items():
        print(f"recognizing texts in {filename}")
        drawing_path = os.path.join(drawing_dir, f"{filename}.jpg")
        if os.path.exists(drawing_path) == True:
            img = cv2.imread(drawing_path)

            for i in range(len(bboxes)):
                print_progress(i,len(bboxes), 'Progress:', 'Complete')
                box_coord = bboxes[i]["bbox"] # [x, y, width, height]
                height = int(box_coord[3] * (1 + text_img_margin_ratio))
                width = int(box_coord[2] * (1 + text_img_margin_ratio))

                x_mid = (box_coord[0] + box_coord[0] + box_coord[2])/2
                x_min = int(x_mid - width/2)
                y_mid = (box_coord[1] + box_coord[1] + box_coord[3]) / 2
                y_min = int(y_mid - height / 2)

                sub_img = img[y_min:y_min + height, x_min:x_min + width, :]

                # if height > width * vertical_threshold: # 세로 문자열로 판단, aspect ratio 기준
                #     sub_img = cv2.rotate(sub_img, cv2.ROTATE_90_CLOCKWISE)

                if bboxes[i]["category_id"] == symbol_dict["text_rotated"]: # 세로 문자열로 판단, "text_rotated 카테고리일경우"
                    sub_img = cv2.rotate(sub_img, cv2.ROTATE_90_CLOCKWISE)



                result_str = pytesseract.image_to_data(sub_img, config="--oem 3 --psm 6")
                recognized_text, conf = parse_tess_result(result_str)

                bboxes[i]["string"] = recognized_text
                bboxes[i]["string_conf"] = conf


                # if height > width * vertical_threshold: # 세로 문자열로 판단
                #     sub_img = cv2.rotate(sub_img, cv2.ROTATE_90_CLOCKWISE)
                #
                # # result_str = pytesseract.image_to_osd(sub_img, config="--oem 3 --psm 6")
                # result_str = pytesseract.image_to_data(sub_img, config="--oem 3 --psm 6")
                # recognized_text, conf = parse_tess_result(result_str)
                #
                # if height < width * vertical_threshold and width < height * vertical_threshold: # 정사각형에 가까운 경우 비교하여 판단
                #     rotated_sub_img = cv2.rotate(sub_img, cv2.ROTATE_90_CLOCKWISE)
                #     rotated_result_str = pytesseract.image_to_data(rotated_sub_img, config="--oem 3 --psm 6")
                #     rotated_recognized_text, rotated_conf = parse_tess_result(rotated_result_str)
                #     if conf < rotated_conf:
                #         recognized_text = rotated_recognized_text
                #         conf = rotated_conf


    return dt_result_text

def parse_tess_result(result_str):
    result_to_list = result_str.split("\n")
    result_string = ""
    confidence = 0
    count = 0
    for result in result_to_list:
        res = result.split("\t")
        if res[0] == '5':
            count += 1
            confidence += int(res[-2])
            result_string = result_string + res[-1]

    if count == 0:
        confidence = 0
    else:
        confidence = confidence / count

    return result_string, confidence

def is_osd_result_rotated(result_str):
    result_to_list = result_str.split("\n")
    result_string = ""
    confidence = 0
    count = 0
    for result in result_to_list:
        if result.find("Rotate:") >= 0:
            res = result.split(":")

            if int(res[1]) == 270 or int(res[1]) == 90:
                return True
            else:
                return False