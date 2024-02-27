from paddleocr import PaddleOCR, draw_ocr
from flask import Flask,request,render_template,jsonify
from PIL import Image
import numpy as np
import cv2
import base64
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def base64_to_cv2(b64str):

    data = base64.b64decode(b64str.encode('utf8'))
    data = np.frombuffer(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data

def recognize_text(img_path):
    ocr = PaddleOCR(det_model_dir='./model/det_db_inference1/', rec_model_dir='./model/en_shuibiao-OCRv4_rec/',
                    rec_char_dict_path='./ppocr/utils/shuibiao_dict.txt', cls_model_dir='./model/cls_inference/',
                    use_angle_cls=True)
    result = ocr.ocr(img_path, cls=True)
    # for idx in range(len(result)):
    #     res = result[idx]
    #     for line in res:
    #         print(line)
    return result



def show_result(img_path, result):
    result = result[0]
    image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores, font_path='/path/to/PaddleOCR/doc/fonts/simfang.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save('result.jpg')



app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        data = request.get_json()
        base64_image = data.get('image', '')
        image = base64_to_cv2(base64_image)
        result = recognize_text(image)
        print(result)
        print(result[0][0][0][0][0])

        processed_data = [
            {"vertices": [{"x": result[0][0][0][0][0], "y": result[0][0][0][0][1]},
                          {"x": result[0][0][0][1][0], "y": result[0][0][0][1][1]},
                          {"x": result[0][0][0][2][0], "y": result[0][0][0][2][1]},
                          {"x": result[0][0][0][3][0], "y": result[0][0][0][3][1]}],
             "label": result[0][0][1][0]},]
        print(processed_data)

        return jsonify(processed_data)

    except Exception as e:
        return jsonify({'error': str(e)})
    return render_template('index.html',)
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000 )