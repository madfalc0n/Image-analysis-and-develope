from flask import Flask, request , jsonify
import json
import urllib
from bs4 import BeautifulSoup
import re
import random

app = Flask(__name__)

class User:
    def __init__(self):
        self.win = 0
        self.lose = 0
        self.draw = 0


user_db = {}
cnt = 0



def result(word): # 가위 바위 보 결과를 회신 0 - 가위 , 1-바위, 2-보 , #사용자-컴퓨터 양수면 사용자 win  음수면 lose, 0이면 draw
    dict_v = { '가위': 0 , '바위':1, '보':2}
    dict_rev = {0:'가위', 1:'바위' , 2:'보'}
    val = random.randint(0,2)
    info = {'me': word , 'bot': dict_rev[val], 'result':''}
    print(info)
    word = dict_v[word]
    
    if word - val == 0:
        info['result'] = 'draw'
        return info
    elif word - val < 0:
        info['result'] = 'lose'
        return info
    elif word - val > 0:
        info['result'] = 'win'
        return info



@app.route('/') 
def home(): 
    return "hello--^^^^---"   


#"session": "projects/mfal-bot1-gowpcm/agent/sessions/e4f5f911-2d0e-98ca-8c22-94f8b924a9b6"
#"session": "projects/mfal-bot1-gowpcm/agent/sessions/02448f02-a4cc-f8d0-59f7-43781f801d8a"

#u = users.get(req[], User())
@app.route('/dialogflow', methods=['POST','GET']) 
def dial():
    global user_db
    print("dialog로 부터 입장")

    req = request.get_json(force=True)# 강제로 json(dict 타입으로) 변환
    #user_db[req['session']] =  user_db.get(req['session'] , User())
    if user_db.get(req['session'] , 0) == 0:
        print("처음 이용자, 초기화 합니다.")
        user_db[req['session']] = User()
    user = user_db[req['session']] 
    #print(type(user_sess))
    print("user session : ", req['session'])
    print(json.dumps(req, indent=4, ensure_ascii=False))

    intentname = req['queryResult']['intent']['displayName']
    print("인텐트 네임", intentname)


    if intentname == 'rps': #단어 찾을때
        print("가위바위보 게임으로 진입합니다")
        print("USER ID: ", req['session'])
        print("전적", user_db[req['session']].win, user_db[req['session']].lose, user_db[req['session']].draw)
        word = req['queryResult']['parameters']['RPS']
        text = result(word)

        if text['result'] == 'win':
            user.win += 1
        elif text['result'] == 'lose':
            user.lose += 1
        else:
            user.draw += 1
        send_m = {'fulfillmentText': f'이긴 횟수: {user.win} , 비긴 횟수: {user.draw}, 진 횟수: {user.lose}'}
        return send_m    
        
    else:
        req = {'fulfillmentText': '제대로보내보라'}
        return jsonify(req)




if __name__ == '__main__':
    app.run(host='0.0.0.0', port =80, debug=True) #debug 모드 True 면 변경사항 바뀌면 알아서 서버가 재실행 됨