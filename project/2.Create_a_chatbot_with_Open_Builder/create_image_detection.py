from flask import Flask, escape, request,send_file, send_from_directory, safe_join, abort
from PIL import Image
import requests

from io import BytesIO
import numpy as np

import matplotlib.pyplot as plt
import cv2

import math
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

app = Flask(__name__)



######이미지용############
@app.route('/' , methods=['POST']) #사용자 정보를 받아옴으로써 POST 사용,  
def image_detect():
    




    # 이미지 JSON GET
    body = request.get_json()
    print(body)

    #URL 파싱 후 이미지 형식으로save
    url = body['userRequest']['params']['media']['url']
    print("출력",body['userRequest']['params']['media']['url'])
    
    
    
    response = requests.get(url)
    img = np.array(Image.open(BytesIO(response.content)))
    img_original = np.array(Image.open(BytesIO(response.content)))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


    img2 = img.copy()
    height, width = img2.shape[:2]
    img_blur = cv2.GaussianBlur(img2, (3,3), 0)
    _, binary =  cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    binary = 255 - binary
    eroded = cv2.morphologyEx(binary, cv2.MORPH_ERODE, (5, 5))
    edged = cv2.Canny(eroded, 10, 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        for i in range(1,11):
            count = 0.01
            approx = cv2.approxPolyDP(c, i*count * peri, True)
            if len(approx ) == 4:
                break
            
    print("approx 값: ", len(approx))
    if len(approx) != 4:
        return {
            "info" : "사각형 인식이 안됨"
        }
    

    def distance(x1, y1, x2, y2):
        result = math.sqrt( math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
        return result


    idx = [1, 0, 2, 3]
    point_list = np.array(approx[idx, 0, :])

    obj_width = distance(point_list[0][0], point_list[0][1], point_list[1][0], point_list[1][1])
    obj_height = distance(point_list[0][0], point_list[0][1], point_list[2][0], point_list[2][1])
    height, width = img.shape[:2]


    pts1 = np.float32([list(point_list[0]), list(point_list[1]), list(point_list[2]), list(point_list[3])])
    pts2 = np.float32([[0, 0], [obj_width, 0], [0, obj_height], [obj_width, obj_height]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img_result = cv2.warpPerspective(img, M, (int(obj_width), int(obj_height)))
    str = pytesseract.image_to_string(img_result)
    print(str)
    # return {
    #     "info" : str
    # }
    return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                        "text": "간단한 텍스트 요소입니다."
                        }},
                    {    "simpleImage": {
                            "imageUrl": "http://734fcb48.ngrok.io/tmp",
                            "altText": "보물상자입니다"
                        }
                    }
                ]
            }
        }



@app.route('/tmp')
def get_image():
    filename = 'tmp/out.png'

    return send_file(filename, mimetype='image/png')



#########테스트용#############
@app.route('/test' , methods=['POST']) #사용자 정보를 받아옴으로써 POST 사용,  
def detect_test():

    #function call
    def img_detec(url):
        response = requests.get(url)
        img = np.array(Image.open(BytesIO(response.content)))
        img_original = np.array(Image.open(BytesIO(response.content)))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img2 = img.copy()
        height, width = img2.shape[:2]
        img_blur = cv2.GaussianBlur(img2, (3,3), 0)
        _, binary =  cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        binary = 255 - binary
        eroded = cv2.morphologyEx(binary, cv2.MORPH_ERODE, (5, 5))
        edged = cv2.Canny(eroded, 10, 250)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            for i in range(1,11):
                count = 0.01
                approx = cv2.approxPolyDP(c, i*count * peri, True)
                if len(approx ) == 4:
                    break
                
        print("approx 값: ", len(approx))
        if len(approx) != 4:
            return {
                "info" : "사각형 인식이 안됨"
            }
        
        def distance(x1, y1, x2, y2):
            result = math.sqrt( math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
            return result

        idx = [1, 0, 2, 3]
        point_list = np.array(approx[idx, 0, :])

        obj_width = distance(point_list[0][0], point_list[0][1], point_list[1][0], point_list[1][1])
        obj_height = distance(point_list[0][0], point_list[0][1], point_list[2][0], point_list[2][1])
        height, width = img.shape[:2]

        pts1 = np.float32([list(point_list[0]), list(point_list[1]), list(point_list[2]), list(point_list[3])])
        pts2 = np.float32([[0, 0], [obj_width, 0], [0, obj_height], [obj_width, obj_height]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img_result = cv2.warpPerspective(img, M, (int(obj_width), int(obj_height)))
        str = pytesseract.image_to_string(img_result)
        print(str)
        print("이미지 검출되었음")
        # return {
        #     "info" : str
        # }
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                        "text": "간단한 텍스트 요소입니다."
                        }},
                    {    "simpleImage": {
                            "imageUrl": "http://734fcb48.ngrok.io/tmp/out.png",
                            "altText": "보물상자입니다"
                        }
                    }
                ]
            }
        }




    # 이미지 JSON GET
    body = request.get_json()
    print(body)

    try:
        #URL 파싱 후 이미지 형식으로save
        url = body['userRequest']['params']['media']['url']
        detect_string = img_detec(url)
        print("출력",body['userRequest']['params']['media']['url'])
        print(detect_string)
        return detect_string
    except KeyError as key:
        print("에러발생", key)
        return {
            "info" : key
        }
    


    








# POST / HTTP/1.1
# Host: your.skill.url
# Accept: application/json
# Content-Type: application/json;charset=UTF-8

# {
#   "intent": {
#     "id": "5a56ec0008cc1461d75291f6", /* block의 id */
#     "name": "스킬테스트" /* block의 이름 */
#   },
#   "userRequest": {
#     "timezone": "Asia/Seoul",
#     "params": {
#       "exampleParam": "example"
#     },
#     "block": {
#       "id": "5a56ec0008cc1461d75291f6",
#       "name": "스킬테스트"
#     },
#     "utterance": "스킬", /* 사용자가 입력한 대화 내용 */
#     "lang": "kr",
#     "user": {
#       "id": "620678", /* 유저의 id 값 */
#       "type": "talk_user_id", /* 유저의 값의 종류  */
#       "properties": { /* 부가적인 아이디 정보들  */
#         "appUserId": "708203191",
#         "appUserStatus": "REGISTERED",
#         "plusfriend_user_key": "BlGTEYoiNoSh"
#       }
#     }
#   },
#   "contexts": [],
#   "bot": {
#     "id": "5a548e36aea1a43fa851ecd9",
#     "name": "또봇"
#   },
#   "action": {
#     "name": "스킬원", /* 스킬의 이름 */
#     "clientExtra": "null", /* button 혹은 바로연결에서 넘겨주는 `extra`의 내용 */
#     "params": {}, /* 스킬 호출시 함께 넘어가는 action parameter */
#     "id": "5a56ebaa211ee046633e958d",
#     "detailParams": {} /* resolve 된 action parameter 내용 */
#   }
# }