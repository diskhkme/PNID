import os
import cv2
from Visualize.image_drawing import draw_bbox_from_bbox_list

mode_option = {1 : "1_GT" , 2 : "2_DT_All", 3 : "3_DT_NMS", 4 : "4_Matching_in_GT", 5 : "5_Matching_in_DT",
             6 : "6_Not_Matching_in_GT", 7 : "7_Not_Matching_in_DT", 8 : "8_Recognized_Text"}
mode_colors = {1 : (255,0,0),2 : (0,0,255),3 : (0,255,0),4 : (255,255,0),5 : (255,255,0),
               6 : (128,128,255),7 : (128,128,255), 8: (0,0,255)}

def draw_test_results_to_img(eval_data, gt_to_dt_match_dict, dt_to_gt_match_dict,
                             drawing_img_dir, output_img_dir, modes, thickness=3):
    """ test 결과를 이미지로 출력하기 위한 함수. 기존과 동일하게 7개의 옵션을 갖고 있음. # TODO : 박스 레이블 표기
        mode 1 : GT 박스 출력
        mode 2 : Detection 결과 출력
        mode 3 : Detection NMS 결과 출력
        mode 4 : GT 박스 중 매칭이 이루어진 박스 출력
        mode 5 : DT 박스 중 매칭이 이루어진 박스 출력
        mode 6 : GT 박스 중 매칭이 이루어지지 않은 박스 출력
        mode 7 : GT 박스 중 매칭이 이루어지지 않은 박스 출력
        mode 8 : Text recognition 결과 출력

    Arguments:
        eval_data (obj): gt_dt_data 객체. 내부에 거의 모든 test 결과를 가지고 있음
        gt_to_dt_match_dict (dict): 도면 이름을 key로, 해당 도면의 gt-dt 매칭 결과를 value로 갖는 dict
        dt_to_gt_match_dict (dict): 도면 이름을 key로, 해당 도면의 dt-gt 매칭 결과를 value로 갖는 dict (위와 동일)
        drawing_img_dir (string): 원본 도면 이미지가 존재하는 폴더
        output_img_dir (string): 결과 이미지 출력 폴더
        modes (list): 1~7까지의 모드 중, 리스트에 숫자가 존재하는 모드를 선택하여 출력
        thickness: 출력 선 두께
    Return:
         None (이미지 파일로 디스크에 저장)
    """
    dt_result = eval_data.dt_result
    gt_json = eval_data.gt_result_json
    dt_result_after_nms = eval_data.dt_result_after_nms

    for img in gt_json['images']:
        print(f"Writing result of {img}...")

        img_filename = img["file_name"]
        img_filename_key = img_filename.split(".")[0]
        image_path = os.path.join(drawing_img_dir, img_filename)
        bboxes_per_image_list = []
        image = cv2.imread(image_path)

        bboxes = gt_json['annotations']
        current_gt_bboxes = [x for x in bboxes if x["image_id"] == img["id"]]
        draw_additional_data = None
        for mode in modes:
            if mode == 1:
                bboxes_per_image = current_gt_bboxes
                draw_additional_data = [x["category_id"] for x in bboxes_per_image]
            elif mode == 2:
                bboxes_per_image = dt_result[img_filename_key]
                draw_additional_data = [x["category_id"] for x in bboxes_per_image]
            elif mode == 3:
                bboxes_per_image = dt_result_after_nms[img_filename_key]
                draw_additional_data = [x["category_id"] for x in bboxes_per_image]
            elif mode == 4:
                bboxes_per_image = [current_gt_bboxes[i] for i in gt_to_dt_match_dict[img_filename_key]]
                draw_additional_data = [x["category_id"] for x in bboxes_per_image]
            elif mode == 5:
                bboxes_per_image = [dt_result_after_nms[img_filename_key][i] for i in dt_to_gt_match_dict[img_filename_key]]
                draw_additional_data = [x["category_id"] for x in bboxes_per_image]
            elif mode == 6:
                gt_num_in_current_img = len(current_gt_bboxes)
                bboxes_per_image = [current_gt_bboxes[i] for i in range(gt_num_in_current_img) if i not in gt_to_dt_match_dict[img_filename_key]]
                draw_additional_data = [x["category_id"] for x in bboxes_per_image]
            elif mode == 7:
                dt_num_in_current_img = len(dt_result_after_nms[img_filename_key])
                bboxes_per_image = [dt_result_after_nms[img_filename_key][i] for i in range(dt_num_in_current_img) if i not in dt_to_gt_match_dict[img_filename_key]]
                draw_additional_data = [x["category_id"] for x in bboxes_per_image]
            elif mode == 8:
                if eval_data.dt_result_text_recognition == None:
                    continue
                text_result = eval_data.dt_result_text_recognition
                bboxes_per_image = text_result[img_filename_key]
                draw_additional_data = [x["string"] for x in text_result[img_filename_key]]
            else:
                print("only supports mode 1 to 8!")

            bboxes = [x["bbox"] for x in bboxes_per_image]
            image_drawed = draw_bbox_from_bbox_list(image, bboxes, draw_additional_data, color=mode_colors[mode], thickness=thickness)
            out_path = os.path.join(output_img_dir, f"{img_filename_key}_{mode_option[mode]}.jpg")
            cv2.imwrite(out_path, image_drawed)
