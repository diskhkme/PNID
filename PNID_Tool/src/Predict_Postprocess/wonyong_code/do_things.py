from json_output_processing import *
from visualize import *
from pathlib import Path


def make_whole_json():
    # xml_dir_path_dataset_0 = '../0. raw_data/original/EWP_Test/xml'
    # xml_dir_path_dataset_1 = '../0. raw_data/resized_0.5/EWP_Test/xml'
    # SymbolClass_txt_file_path = '../0. raw_data/EWP_SymbolClass_sym_only.txt'
    #
    # whole_image_gt_json = make_whole_image_gt_json(xml_dir_path_dataset_0, SymbolClass_txt_file_path)
    # with open('./whole_image_json/gt_json/{}.json'.format('dataset_0_test_whole_image'), 'w', encoding="utf-8") as f:
    #     json.dump(whole_image_gt_json, f, ensure_ascii=False, indent=2)
    #
    # whole_image_gt_json = make_whole_image_gt_json(xml_dir_path_dataset_1, SymbolClass_txt_file_path)
    # with open('./whole_image_json/gt_json/{}.json'.format('dataset_1_test_whole_image'), 'w', encoding="utf-8") as f:
    #     json.dump(whole_image_gt_json, f, ensure_ascii=False, indent=2)

    # result json 통합
    gt_json_path = './gt_json/dataset_1_test.json'
    gt_whole_json_path = './whole_image_json/gt_json/dataset_1_test_whole_image.json'
    dataset_json_path = './result_json/experiment_6_gfl_dataset_1_adam.bbox.json'

    with open(gt_json_path) as f:
        gt_json = json.load(f)

    with open(dataset_json_path) as f:
        result_json = json.load(f)

    with open(gt_whole_json_path) as f:
        whole_image_gt_json = json.load(f)

    whole_image_result_json = make_whole_image_result_json(whole_image_gt_json, gt_json, result_json)
    with open('./whole_image_json/result_json/{}_whole_image.bbox.json'.format('experiment_6_gfl_dataset_1_adam'), 'w', encoding="utf-8") as f:
        json.dump(whole_image_result_json, f, ensure_ascii=False, indent=2)


def visualization(mode, output_dir_name, whole_gt_json_path, whole_result_json_path):
    if mode == 0:
        test_dir = 'original'
    elif mode == 1:
        test_dir = 'resized_0.5'
    else:
        raise

    # 1. gt
    with open(whole_gt_json_path) as f:
        whole_img_gt = json.load(f)
    draw_bbox_from_whole_img_gt_json('../0. raw_data/{}/EWP_Test/image'.format(test_dir), '{}/1.GT'.format(output_dir_name), whole_img_gt,
                                     color=(128, 128, 128))

    # 2. result
    with open(whole_result_json_path) as f:
        whole_img_result = json.load(f)
    draw_bbox_from_whole_img_result_json('../0. raw_data/{}/EWP_Test/image'.format(test_dir), '{}/2.Result_all'.format(output_dir_name), whole_img_gt, whole_img_result,
                                         color=(48, 48, 48))

    image_name_to_GT_bboxs_dict = process_whole_image_gt_json(whole_img_gt)
    image_name_to_Result_bboxs_dict = process_whole_image_result_json(whole_img_gt, whole_img_result)

    # 3. result after nms
    after_nms_save_dir = Path('./{}/3.Result_after_nms'.format(output_dir_name))
    after_nms_save_dir.mkdir(parents=True, exist_ok=True)
    for image_name, bboxes in image_name_to_Result_bboxs_dict.items():
        image_path = '../0. raw_data/{}/EWP_Test/image/{}.jpg'.format(test_dir, image_name)
        output_path = str(after_nms_save_dir.joinpath('{}.jpg'.format(image_name)).resolve())

        image = cv2.imread(image_path)

        after_nms = non_max_suppression_fast(bboxes, 0.0)
        image = draw_bbox_from_bbox_list(image, after_nms, color=(255, 0, 0), thickness=6)

        cv2.imwrite(output_path, image)

    # 4. detected symbol GT
    detected_symbol_GT = Path('./{}/4.detected_sym_GT'.format(output_dir_name))
    detected_symbol_GT.mkdir(parents=True, exist_ok=True)
    for image_name in image_name_to_GT_bboxs_dict.keys():
        image_path = '../0. raw_data/{}/EWP_Test/image/{}.jpg'.format(test_dir, image_name)
        output_path = str(detected_symbol_GT.joinpath('{}.jpg'.format(image_name)).resolve())

        gt_bboxes = image_name_to_GT_bboxs_dict[image_name]
        result_bboxes = image_name_to_Result_bboxs_dict[image_name]
        after_nms_result_bboxes = non_max_suppression_fast(result_bboxes, 0.0)

        gt_to_result_index_match_dict, result_to_gt_index_match_dict = compare_gt_and_result(gt_bboxes,
                                                                                             after_nms_result_bboxes)
        image = cv2.imread(image_path)
        for gt_index, result_index in gt_to_result_index_match_dict.items():
            gt_box = gt_bboxes[gt_index]

            x1, y1, x2, y2 = gt_box[0], gt_box[1], gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]
            cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=6)

        cv2.imwrite(output_path, image)

    # 5. detected symbol Result
    detected_symbol_Result = Path('./{}/5.detected_sym_Result'.format(output_dir_name))
    detected_symbol_Result.mkdir(parents=True, exist_ok=True)
    for image_name in image_name_to_GT_bboxs_dict.keys():
        image_path = '../0. raw_data/{}/EWP_Test/image/{}.jpg'.format(test_dir, image_name)
        output_path = str(detected_symbol_Result.joinpath('{}.jpg'.format(image_name)).resolve())

        gt_bboxes = image_name_to_GT_bboxs_dict[image_name]
        result_bboxes = image_name_to_Result_bboxs_dict[image_name]
        after_nms_result_bboxes = non_max_suppression_fast(result_bboxes, 0.0)

        gt_to_result_index_match_dict, result_to_gt_index_match_dict = compare_gt_and_result(gt_bboxes,
                                                                                             after_nms_result_bboxes)
        image = cv2.imread(image_path)
        for gt_index, result_index in gt_to_result_index_match_dict.items():
            result_box = after_nms_result_bboxes[result_index]

            x1, y1, x2, y2 = int(result_box[0]), int(result_box[1]), int(result_box[0] + result_box[2]), int(result_box[1] + result_box[3])
            cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=6)

        cv2.imwrite(output_path, image)

    # 6. not detected symbol GT
    not_detected_sym_GT = Path('./{}/6.not_detected_sym_GT'.format(output_dir_name))
    not_detected_sym_GT.mkdir(parents=True, exist_ok=True)
    for image_name in image_name_to_GT_bboxs_dict.keys():
        image_path = '../0. raw_data/{}/EWP_Test/image/{}.jpg'.format(test_dir, image_name)
        output_path = str(not_detected_sym_GT.joinpath('{}.jpg'.format(image_name)).resolve())

        gt_bboxes = image_name_to_GT_bboxs_dict[image_name]
        result_bboxes = image_name_to_Result_bboxs_dict[image_name]
        after_nms_result_bboxes = non_max_suppression_fast(result_bboxes, 0.0)

        gt_to_result_index_match_dict, result_to_gt_index_match_dict = compare_gt_and_result(gt_bboxes,
                                                                                             after_nms_result_bboxes)
        image = cv2.imread(image_path)
        for gt_index, gt_box in enumerate(gt_bboxes):
            if gt_index not in gt_to_result_index_match_dict.keys():
                x1, y1, x2, y2 = gt_box[0], gt_box[1], gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]
                cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=6)
        cv2.imwrite(output_path, image)

    # 7. not detected symbol Result
    not_detected_sym_Result = Path('./{}/7.not_detected_sym_Result'.format(output_dir_name))
    not_detected_sym_Result.mkdir(parents=True, exist_ok=True)
    for image_name in image_name_to_GT_bboxs_dict.keys():
        image_path = '../0. raw_data/{}/EWP_Test/image/{}.jpg'.format(test_dir, image_name)
        output_path = str(not_detected_sym_Result.joinpath('{}.jpg'.format(image_name)).resolve())

        gt_bboxes = image_name_to_GT_bboxs_dict[image_name]
        result_bboxes = image_name_to_Result_bboxs_dict[image_name]
        after_nms_result_bboxes = non_max_suppression_fast(result_bboxes, 0.0)

        gt_to_result_index_match_dict, result_to_gt_index_match_dict = compare_gt_and_result(gt_bboxes,
                                                                                             after_nms_result_bboxes)

        image = cv2.imread(image_path)
        for result_index, result_box in enumerate(after_nms_result_bboxes):
            if result_index not in result_to_gt_index_match_dict.keys():
                x1, y1, x2, y2 = int(result_box[0]), int(result_box[1]), int(result_box[0] + result_box[2]), int(
                    result_box[1] + result_box[3])
                cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=6)

        cv2.imwrite(output_path, image)

    # result summary
    for image_name in image_name_to_GT_bboxs_dict.keys():

        gt_bboxes = image_name_to_GT_bboxs_dict[image_name]
        result_bboxes = image_name_to_Result_bboxs_dict[image_name]
        after_nms_result_bboxes = non_max_suppression_fast(result_bboxes, 0.0)

        gt_to_result_index_match_dict, result_to_gt_index_match_dict = compare_gt_and_result(gt_bboxes,
                                                                                             after_nms_result_bboxes)
        gt_class_num_dict = defaultdict(int)
        gt_detected_class_num_dict = defaultdict(int)

        gt_num = 0
        detected_num = 0
        all_prediction = len(after_nms_result_bboxes)
        for i in after_nms_result_bboxes:
            all_prediction += 1

        for gt_bbox in gt_bboxes:
            gt_class_num_dict[gt_class] += 1
            gt_num += 1

        for gt_index in gt_to_result_index_match_dict.keys():
            gt_bbox = gt_bboxes[gt_index]
            gt_class = gt_bbox[4]
            gt_detected_class_num_dict[gt_class] += 1
            detected_num += 1

        with open('./{}/{}.txt'.format(output_dir_name, image_name), 'w+') as f:
            print(image_name)
            f.write(image_name + '\n')

            gt_classes = list(gt_class_num_dict.keys())
            gt_classes.sort()
            for class_index in gt_classes:
                gt = gt_class_num_dict[class_index]
                detected = gt_detected_class_num_dict[class_index]
                print('class {} : detected / gt = {} / {} = {}'.format(class_index, detected, gt, detected / gt))
                f.write('class {} : detected / gt = {} / {} = {}'.format(class_index, detected, gt, detected / gt) + '\n')
            print('precision = {} / {} = {}  recall = {} / {} = {}'.format(detected_num, all_prediction, detected_num / all_prediction,
                                                                            detected_num, gt_num, detected_num / gt_num))
            f.write('precision = {} / {} = {}  recall = {} / {} = {}'.format(detected_num, all_prediction, detected_num / all_prediction,
                                                                            detected_num, gt_num, detected_num / gt_num) + '\n')

            print()


if __name__ == '__main__':
    visualization(mode=1, output_dir_name='dataset_0_aaaaa',
                  whole_gt_json_path='//Predict_Postprocess/test_gt_global.json',
                  whole_result_json_path='//Predict_Postprocess/test_dt_global.json')

