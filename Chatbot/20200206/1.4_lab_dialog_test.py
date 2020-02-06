from flask import Flask, request , jsonify
import json
import urllib
from bs4 import BeautifulSoup

app = Flask(__name__)


cnt = 0
def getmean(word):
    url = "https://search.naver.com/search.naver?where=kdic&query="
    url = url + urllib.parse.quote_plus(word)
    bs = BeautifulSoup(urllib.request.urlopen(url).read(), "html.parser") 
    output = bs.select('p.txt_box')
    return [node.text  for node in output     ] 

def getWeather(city) :    
    url = "https://search.naver.com/search.naver?query="
    url = url + urllib.parse.quote_plus(city + "날씨")
    print(url)
    bs = BeautifulSoup(urllib.request.urlopen(url).read(), "html.parser")
    temp = bs.select('span.todaytemp')    
    desc = bs.select('p.cast_txt')    
    return  {"temp":temp[0].text, "desc":desc[0].text}   





@app.route('/') 
def home(): # 쿼리 예시 localhost/?name=fuck&item=hol
    name = request.args.get("name")    
    item = request.args.get("item")    
    return "hello--^^^^---" + name + item   



@app.route('/abc')
def abc():
    return "test~~~~~~~"



@app.route('/dialogflow', methods=['POST','GET']) 
def dial2():
    print("dialog로 부터 입장")
    # if request.method == "GET": #GET 방식일 경우
    #     #city = request.args.get('city') #무조건 GET 방식으로 불러온다, POST 방식으로 요청오는건 못받음, 장점은 값이 없으면 None으로 받아옴
    #     req = request.args # dict 로 리턴
    #     print("GET 방식 진행됨" , req)
    #     return "겟방식"

    req = request.get_json(force=True)# 강제로 json(dict 타입으로) 변환
    print(json.dumps(req, indent=4, ensure_ascii=False))
    # j_file = 'req_json.json'
    # with open(j_file,'wr', encoding='UTF-8' ) as json_file: #json 파일로 저장
    #     j_req = json.load(json_file)
    #     print(json.dumps(j_req, indent=4, ensure_ascii=False)) #출력시 한글이 깨지지 않도록 설정


    intentname = req['queryResult']['intent']['displayName']
    
    if intentname == 'dict': #단어 찾을때
        word = req['queryResult']['parameters']['any']
        text = getmean(word)[0]
        text = {'fulfillmentText': text}
        print(text)
        return jsonify(text)
    elif intentname == 'order2': #음식 주문할 때
        foodname = req['queryResult']['parameters']['food_number'][0]['foodname']
        foodnumber = int(req['queryResult']['parameters']['food_number'][0]['foodnumber'])
        text = foodname + ' ' + str(foodnumber) +'개요~'
        return {'fulfillmentText': text}

    elif intentname == 'weather': #날씨 물어볼 때
        city = req['queryResult']['parameters']['geo-city']
        date = req['queryResult']['parameters']['date']
        info = getWeather(city)
        temp = info['temp']
        desc = info['desc'] 
        text = '오늘 날씨는 '+ temp+'도이고, '+desc
        return {'fulfillmentText': text}
    
    else:
        req = {'fulfillmentText': 'Hello~~'}
        return jsonify(req)








if __name__ == '__main__':
    app.run(host='0.0.0.0', port =80, debug=True) #debug 모드 True 면 변경사항 바뀌면 알아서 서버가 재실행 됨