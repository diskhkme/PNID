import cv2

def draw_bbox_from_bbox_list(image, bbox_list, box_data, color, thickness):
    """ box_list의 박스들을 image에 그리고 반환
    """
    if box_data is not None:
        assert len(bbox_list) == len(box_data)

    image_ = image.copy()
    for i in range(len(bbox_list)):
        x1, y1, w, h = bbox_list[i][:4]
        x1, y1, x2, y2 = int(x1), int(y1), int(x1 + w), int(y1 + h)
        cv2.rectangle(image_, (x1, y1), (x2, y2), color=color, thickness=thickness)

        if box_data is not None:
            cv2.putText(image_,box_data[i],(x1,y1), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color)

    return image_