from flask import Flask, escape, request, send_file, send_from_directory, redirect, url_for, g
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import json
import re
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract' #tesseract가 설치된 경로, 인터넷에서 설치법을 참고하여 설치한다.

app = Flask(__name__)



id = 0
db = {}



#개별명함조회
@app.route('/view_per_namecard', methods=['GET','POST'])
def view_per_namecard():
    print("개인명함 조회하는 함수")
    body = request.get_json()
    print(body)
    print("body['userRequest']['utterance'] : " , body['userRequest']['utterance'])
    number = re.findall("\d+",body['userRequest']['utterance'])
    print("숫자추출 번호: ",number)
    email = db[number[0]]['email']
    p_numb = db[number[0]]['number']
    return {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": "1. 이메일 정보: \n"+email+"\r\n2. 전화번호: \n"+p_numb
                    }
                }
            ]
        }
    }

#명함전체조회
@app.route('/viewnamecard', methods=['GET','POST'])
def view_namecard():
    global id
    if id == 0 :
        return {
                "version": "2.0",
                "template": {
                    "outputs": [
                        {
                            "simpleText": {
                                "text": "명함이 저장되어 있지 않습니다. 명함을 넣어주세요."
                            }
                        }
                    ]
                }
            }
    
    tmp_str = '['
    for i in range(1,id+1):
        n_title = str(i)+"번째 명함"
        n_description = str(i)+"번째 명함 입니다."
        n_imageUrl = "http://bb155e2b.ngrok.io/tmp/result_"+str(i)+".png"
        tmp_str += """{ "title": " """+str(n_title)+""" ", "description": " """+str(n_description)+""" ","thumbnail": {"imageUrl": " """+n_imageUrl+""" "},"buttons": [{"action": "message","label": "명함정보 보기","messageText": " """+str(i)+""" 번명함보기"}]}"""
        if i != id:
            tmp_str += ','
        elif i == id:
            tmp_str += ']'
    new_tmp_str  = tmp_str.replace(" ", "")
    #print(tmp_str)
    json_string = """{
        "version": "2.0",
        "template": {
            "outputs": [
            {
                "carousel": {
                "type": "basicCard",
                "items": """+new_tmp_str+"""
                
                }
            }
            ]
        }
        }"""
    print(json_string)
    json_val = json.loads(json_string)
    print(json_val)
    return json_val


#OCR 결과 호출
@app.route('/tmp/<result>.png')
def img_file_download(result):
    file_name = f"tmp/"+result+".png"
    return send_file(file_name,
                     mimetype='image/png',
                     attachment_filename=result+'.png',# 다운받아지는 파일 이름. 
                     as_attachment=True)


#########서비스용#############
@app.route('/main' , methods=['POST']) #사용자 정보를 받아옴으로써 POST 사용,  
def detect_test():
    #function call
    def img_detec(url):
        #명함검출 및 문장 인식 부분
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
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            for i in range(1,11):
                count = 0.01
                approx = cv2.approxPolyDP(c, i*count * peri, True)
                if len(approx ) == 4:
                    break


        print("approx 값: ", len(approx))
        if len(approx) != 4:
            binary = 255 + binary
            eroded = cv2.morphologyEx(binary, cv2.MORPH_ERODE, (5, 5))
            edged = cv2.Canny(eroded, 10, 250)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
            cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #maxArea = 0
            for c in cnts:
                area = cv2.contourArea(c)
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.03 * peri, True)
                for i in range(1,11):
                    count = 0.01
                    approx = cv2.approxPolyDP(c, i*count * peri, True)
                    if len(approx ) == 4:
                        break



            if len(approx) != 4:
                return {
                    "version": "2.0",
                    "template": {
                        "outputs": [
                            {
                                "simpleText": {
                                    "text": "사각형 인식이 안됩니다. 제대로 넣어주세요"
                                }
                            }
                        ]
                    }
                }
        
        def distance(x1, y1, x2, y2):
            result = math.sqrt( math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
            return result

        if approx[2][0][0]<approx[0][0][0] :
            idx = [1, 0, 2, 3]
        else:
            idx = [0,3,1,2]

        #idx = [1, 0, 2, 3]
        
        point_list = np.array(approx[idx, 0, :])

        obj_width = distance(point_list[0][0], point_list[0][1], point_list[1][0], point_list[1][1])
        obj_height = distance(point_list[0][0], point_list[0][1], point_list[2][0], point_list[2][1])
        height, width = img.shape[:2]

        pts1 = np.float32([list(point_list[0]), list(point_list[1]), list(point_list[2]), list(point_list[3])])
        pts2 = np.float32([[0, 0], [obj_width, 0], [0, obj_height], [obj_width, obj_height]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img_result = cv2.warpPerspective(img, M, (int(obj_width), int(obj_height)))

        global id 
        id += 1
        #이미지 저장 및 id 추가
        img_name = 'result_'+str(id)
        cv2.imwrite("./tmp/"+img_name+".png" ,img_result)
        

        #이미지에서 검출된 문자열 저장
        img_to_str = pytesseract.image_to_string(img_result, 'eng+kor')
        print(img_to_str)

        #db에 데이터 넣기
        global db
        content = re.sub("[~]", "", img_to_str)
        num = []
        nan = []
        regex_1 = re.compile(r'\d{3,4} \d+-\d+')
        regex_01 = re.compile(r'\d{3,4} \d+ \d+')
        regex_001 = re.compile(r'\d{3,4}-\d+-\d+')
        if regex_1.findall(content) != nan:
            num.append(regex_1.findall(content))
        elif regex_01.findall(content) != nan:
            num.append(regex_01.findall(content))
        elif regex_001.findall(content) != nan:
            num.append(regex_001.findall(content))
        regex_2 = re.compile(r'.*@.*')
        email = regex_2.findall(content)
        print(email, num)
        try:
            email = email[0]
        except IndexError as idxer:
            print("인덱스 에러 발생",idxer)
            email = '검출이 안됨'

        try:
            num = num[0][0]
        except IndexError as idxer2:
            print("인덱스 에러 발생2",idxer2)
            num = '검출이 안됨'

        # email = email[0]
        # num = num[0][0]
        db[str(id)] = {
            "id" : id,
            "number" : str(num),
            "email" : str(email)
        }
        print("db 정보 조회 \n",db)
        #print(img_to_str)
        print("이미지 검출되었음")
        json_string = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                        "text": "명함인식 결과: \n"+img_to_str
                        }},
                    {    "simpleImage": {
                            "imageUrl": "http://bb155e2b.ngrok.io/tmp/"+img_name+".png",
                            "altText": "명함"
                        }
                    }
                ]
            }
        }
        return json_string


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
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "명함을 올려주세요."
                        }
                    }
                ]
            }
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