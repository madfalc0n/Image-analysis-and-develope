import requests
import time
import sys
import os
import hashlib
import hmac
import base64
import json
from datetime import datetime
import argparse

"""
Naver Cloud Platform SMS API를 사용
"""
# cur_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
# print(cur_time)

# NCP를 사용하기 위한 변수 설정
url = "https://sens.apigw.ntruss.com"
requestUrl = "/sms/v2/services/"
serviceId = "yourserviceid" # NCP에서 발급받은 service id
requestUrl2 = "/messages" 
access_key = "youraccesskey" # NCP에서 발급받은 access key
secret_key = "yoursecretkey" # NCP에서 발급받은 accesskey에 매핑되는 secret key

# 위 변수를 합쳐 uri로 저장
# uri = /sms/v2/services/yourserviceid/messages
uri = requestUrl + serviceId + requestUrl2
# API를 호출하기위한 변수 저장
# apiUrl = https://sens.apigw.ntruss.com/sms/v2/services/yourserviceid/messages
apiUrl = url + uri

# API요청시 헤더에 추가할 시그니처 키 생성함수
def make_signature(uri, timestamp, access_key, secret_key):
    # secret_key값을 UTF-8 형식의 바이트 타입으로 secret_key 변수에 저장
    secret_key = bytes(secret_key, 'UTF-8')
    
    # POST method 사용
    method = "POST"

    # msessage = POST /sms/v2/services/yourserviceid/messages\n{timestamp}\n{access_key}
    message = method + " " + uri + "\n" + timestamp + "\n" + access_key
    # msessage 값을 UTF-8 형식의 바이트 타입으로 msessage 변수에 저장
    message = bytes(message, 'UTF-8')
    
    # 바이트 타입의 객체를 hash sha256 방식으로 암호화 및 인코딩 후 리턴 
    signingKey = base64.b64encode(hmac.new(secret_key, message, digestmod=hashlib.sha256).digest())
    return signingKey

# NCP에 요청을 보내기위한 json 형식의 변수 생성 및 반환 
def create_data(send_phonenum, recv_phonenum, cur_time):
    msg = cur_time + '\n' + "응급환자 발생! 확인 요망"
    data = {
        "type": "SMS",
        "contentType": "COMM",
        "countryCode": "82",
        "from": send_phonenum,
        "content": "message",
        "messages": [
            {
                "to": recv_phonenum,
                "content": msg
            }
        ]
    }
    return data

# 객체인식후 응급환자 발생 시 호출되는 함수
# NCP에서 요구하는 값(헤더,시그니처)을 포함시켜 관리자에게 SMS 메시지를 전송
def send_msg(send_phonenum, recv_phonenum, cur_time):
    print(f"send : {send_phonenum}, recv : {recv_phonenum}, cur_time : {cur_time}")
    result = []

    # 시그니쳐 생성
    timestamp = int(time.time() * 1000)
    timestamp = str(timestamp)
    # print(f"타임스탬프 값: {timestamp}")
    signature = make_signature(uri, timestamp, access_key, secret_key)
    print(signature)

    # 헤더 생성
    headers = {
        'Content-Type': 'application/json; charset=utf-8',
        'X-ncp-apigw-timestamp': timestamp,
        'x-ncp-iam-access-key': access_key,
        'x-ncp-apigw-signature-v2': signature
    }

    # SMS 보내기
    for user in recv_phonenum:
        data = create_data(send_phonenum, user, cur_time)
        data2 = json.dumps(data)
        response = requests.post(apiUrl, headers=headers, data=data2)
        result.append(response.text.encode('utf8'))
        time.sleep(0.03)  # 외부와 통신하므로 혹시모르니 딜레이 줌

    # 결과 확인
    # print("Send Message result : ",result)
    return result

# 테스트용으로 python send_sms.py를 실행할 경우 아래 조건문 실행됨
# 상위 폴더의 main.py에서 실행할 경우 아래 조건문은 실행되지 않음
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sender', type=str, default='01036900941', help='Send phone number')
    parser.add_argument('-r', '--receiver', type=str, required=True, help='Receive phone number')

    args = parser.parse_args()
    send_phonenum = args.sender
    recv_phonenum = args.receiver
    cur_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print(f"send : {send_phonenum}, recv : {recv_phonenum}, cur_time : {cur_time}")
    result = []

    # 시그니쳐 생성
    timestamp = int(time.time() * 1000)
    timestamp = str(timestamp)
    # print(f"타임스탬프 값: {timestamp}")
    signature = make_signature(uri, timestamp, access_key, secret_key)
    print(signature)

    # 헤더 생성
    headers = {
        'Content-Type': 'application/json; charset=utf-8',
        'X-ncp-apigw-timestamp': timestamp,
        'x-ncp-iam-access-key': access_key,
        'x-ncp-apigw-signature-v2': signature
    }

    # SMS 보내기
    data = create_data(send_phonenum, recv_phonenum, cur_time)
    data2 = json.dumps(data)
    response = requests.post(apiUrl, headers=headers, data=data2)
    result.append(response.text.encode('utf8'))
    print(result)