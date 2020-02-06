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


@app.route('/')
def home():
    global cnt
    cnt += 1
    return f"""
    <h1> Hello </h1> <br>
    {cnt}명이 방문 했습니다. <br>
    <iframe height="430" width="350" src="https://bot.dialogflow.com/madfalconweb"></iframe>    
    """

# 개발은 실제(단위) 테스트 할떄 GET POST 둘다 진행함
@app.route('/weather', methods=['POST','GET'])
def weather():
    if request.method == "POST": #POST 방식일 경우
        #city = request.form.get('city', 'default값') # POST 방식으로 요청시 받아옴, 단점은 값이 없으면 뻗어버림, 뒤에 default 값을 넣어주면 값을 안적었을 경우 default 값으로 리턴
        req = request.form # dict 로 리턴
        print("POST 방식 진행됨")

    elif request.method == "GET": #GET 방식일 경우
        #city = request.args.get('city') #무조건 GET 방식으로 불러온다, POST 방식으로 요청오는건 못받음, 장점은 값이 없으면 None으로 받아옴
        req = request.args # dict 로 리턴
        print("GET 방식 진행됨")
    
    #삼항 연산자
    #req = request.args if request.method == 'GET' else request.form
    
    city = req.get('city') 
    
    return f"{city} 날씨 좋아요"




# @app.route('/dialogflow', methods=['POST','GET']) 
# def dial():
#     # if request.method == "GET": #GET 방식일 경우
#     #     #city = request.args.get('city') #무조건 GET 방식으로 불러온다, POST 방식으로 요청오는건 못받음, 장점은 값이 없으면 None으로 받아옴
#     #     req = request.args # dict 로 리턴
#     #     print("GET 방식 진행됨" , req)
#     #     return "겟방식"

#     req = request.get_json(force=True)# 강제로 json(dict 타입으로) 변환
#     print(json.dumps(req, indent=4, ensure_ascii=False)) #출력시 한글이 깨지지 않도록 설정
#     req = {'fulfillmentText': 'Hello~~'}
#     return jsonify(req)


@app.route('/dialogflow', methods=['POST','GET']) 
def dial2():
    print("dialog로 부터 입장")
    # if request.method == "GET": #GET 방식일 경우
    #     #city = request.args.get('city') #무조건 GET 방식으로 불러온다, POST 방식으로 요청오는건 못받음, 장점은 값이 없으면 None으로 받아옴
    #     req = request.args # dict 로 리턴
    #     print("GET 방식 진행됨" , req)
    #     return "겟방식"

    req = request.get_json(force=True)# 강제로 json(dict 타입으로) 변환
    print(json.dumps(req, indent=4, ensure_ascii=False)) #출력시 한글이 깨지지 않도록 설정

    #answer = req['result']['fulfillment']['speech']
    intentname = req['queryResult']['intent']['displayName']

    if intentname == 'dict':
        word = req['queryResult']['parameters']['any']
        text = getmean(word)[0]
        text = {'fulfillmentText': text}
        print(text)
        return jsonify(text)
    else:
        req = {'fulfillmentText': 'Hello~~'}
        return jsonify(req)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port =80, debug=True) #debug 모드 True 면 변경사항 바뀌면 알아서 서버가 재실행 됨