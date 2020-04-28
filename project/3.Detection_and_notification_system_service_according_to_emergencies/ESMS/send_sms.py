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
Naver Cloud Platform SMS API
"""
# cur_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
# print(cur_time)

url = "https://sens.apigw.ntruss.com"
requestUrl = "/sms/v2/services/"
serviceId = "yourserviceid"
requestUrl2 = "/messages"
access_key = "youraccesskey"
secret_key = "yoursecretkey"
uri = requestUrl + serviceId + requestUrl2
apiUrl = url + uri


def make_signature(uri, timestamp, access_key, secret_key):
    secret_key = bytes(secret_key, 'UTF-8')

    method = "POST"
    message = method + " " + uri + "\n" + timestamp + "\n" + access_key
    message = bytes(message, 'UTF-8')
    signingKey = base64.b64encode(hmac.new(secret_key, message, digestmod=hashlib.sha256).digest())
    return signingKey


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