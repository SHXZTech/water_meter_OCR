from paddleocr import PaddleOCR, draw_ocr
import math
import cv2
import sys
import os
import numpy as np

corepath = os.path.abspath("../ultralytics")
sys.path.append(corepath)
from ultralytics import YOLO

# print(sys.path)


def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def angle_between_points(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    if x1 == x2:
        return 90
    elif y1 == y2:
        return 0
    else:
        angle = math.degrees(math.atan((y2 - y1) / (x2 - x1)))
        return angle




def rotation(img, degree):
    # degree左转
    height, width = img.shape[:2]
    heightNew = int(
        width * math.fabs(math.sin(math.radians(degree))) + height * math.fabs(math.cos(math.radians(degree))))
    widthNew = int(
        height * math.fabs(math.sin(math.radians(degree))) + width * math.fabs(math.cos(math.radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    return imgRotation


def recognize_text(img_path):
    ocr = PaddleOCR(det_model_dir='./model/det_db_inference1/', rec_model_dir='./model/en_shuibiao-OCRv4_rec/',
                    rec_char_dict_path='./ppocr/utils/shuibiao_dict.txt', cls_model_dir='./model/cls_inference/',
                    use_angle_cls=True)
    result = ocr.ocr(img_path, cls=True)
    return result


def process(img):
    result = recognize_text(img)
    point0 = (result[0][0][0][0][0], result[0][0][0][0][1])
    point1 = (result[0][0][0][1][0], result[0][0][0][1][1])
    point2 = (result[0][0][0][2][0], result[0][0][0][2][1])
    point3 = (result[0][0][0][3][0], result[0][0][0][3][1])
    s1 = calculate_distance(point0, point1)
    s2 = calculate_distance(point0, point3)
    if s1 > s2:
        angle = angle_between_points(point0, point1)
    else:
        angle = angle_between_points(point0, point3)
    # print(angle)

    img = rotation(img, angle)
    model = YOLO('detect/best.pt')
    res = model.predict(img, save=False,nms=True,agnostic_nms=True)
    li = []
    l2 = []
    for re in res[0]:
        if re.boxes.cls.tolist()[0] != 10:
            li.append({'x': int(re.boxes.xyxy.tolist()[0][0]), 'y': int(re.boxes.xyxy.tolist()[0][1]),
                       'cls': int(re.boxes.cls.tolist()[0]),'conf':re.boxes.conf.tolist()[0]})
        else:
            l2.append(int(re.boxes.xyxy.tolist()[0][1]))
    # print(li)
    l1 = sorted(li, key=lambda x: x['x'], reverse=True)
    num = str(result[0][0][1][0])
    num1="."
    # print('l1len',l1.__len__())
    if l1.__len__() > 0:
        if l2.__len__() > 0:
            if l2[0] < l1[0]['y']:
                # for i, data in enumerate(l1):
                #     num = num + data['cls'] * 10 ** (-i - 1)
                for data in l1:
                    num1 = num1 + str(data['cls'])

            else:
                # for i, data in enumerate(l1):
                #     num = num + (data['cls'] + 5) % 10 * 10 ** (-(l1.__len__() - 1 - i) - 1)
                l1.reverse()
                for data in l1:
                    num1 = num1 + str((data['cls'] + 5) % 10)

        else:
            for data in l1:
                num1 = num1 + str(data['cls'])

    num = num+num1
    processed_data = [
        {"vertices": [{"x": result[0][0][0][0][0], "y": result[0][0][0][0][1]},
                      {"x": result[0][0][0][1][0], "y": result[0][0][0][1][1]},
                      {"x": result[0][0][0][2][0], "y": result[0][0][0][2][1]},
                      {"x": result[0][0][0][3][0], "y": result[0][0][0][3][1]}],
         "label": str(num), }]
    return processed_data


if __name__ == '__main__':
    path = 'E:\\test\\watermeter\\0007.jpg'
    path = cv2.imread(path)
    processed_data = process(path)
    print(processed_data)
