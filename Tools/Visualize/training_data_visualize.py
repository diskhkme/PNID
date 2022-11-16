import os
import cv2
import json
from pathlib import Path
from image_drawing import draw_bbox_from_bbox_list

def draw_training_data_bbox(data_path: str, json_path: str, output_img_dir: str):
    with open(json_path, 'r') as js:
        data = json.load(js);
        bboxes = data['annotations']
    
        for img_dict in data['images']:
            img_filename = img_dict["file_name"]
            img_filename_key = img_filename.split(".")[0]
            img_path = os.path.join(data_path, img_filename)
            image = cv2.imread(img_path)
            
            bboxes_per_image = [x for x in bboxes if x["image_id"] == img_dict["id"]]
            draw_additional_data = [x["category_id"] for x in bboxes_per_image]

            bboxes_ = [x["bbox"] for x in bboxes_per_image]
            image_drawed = draw_bbox_from_bbox_list(image, bboxes_, draw_additional_data, color=(255,0,0), thickness=3)
            out_path = os.path.join(output_img_dir, f"{img_filename_key}_0_GT.jpg")
            cv2.imwrite(out_path, image_drawed)

if __name__=='__main__':
    data_path = 'C:\\Users\\DongwonJeong\\Desktop\\HyundaiPNID\\Dataset\\Dataset_Big_0.125\\train\\'
    json_path = 'C:\\Users\\DongwonJeong\\Desktop\\HyundaiPNID\\Dataset\\Dataset_Big_0.125\\train.json'
    drawing_dir = "C:\\Users\\DongwonJeong\\Desktop\\HyundaiPNID\\Dataset\\Dataset_Big_0.125\\visualize\\"

    draw_training_data_bbox(data_path, json_path, drawing_dir)