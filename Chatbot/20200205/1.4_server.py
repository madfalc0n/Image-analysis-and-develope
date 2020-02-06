from flask import Flask, request, jsonify
import requests
import json
import urllib
from bs4 import BeautifulSoup





def getweather(city):
    #인코딩 처리 해주는 함수 urllib.parse.quote_plus(city+"날씨")
    url = "https://search.naver.com/search.naver?&query="
    url = url + urllib.parse.quote_plus(city+"날씨")
    print("city : ",city)
    print("url 호출: ", url)
    bs = BeautifulSoup(urllib.request.urlopen(url).read(), "html.parser")

    temp = bs.findAll('span', 'todaytemp') # 태그, 클래스
    desc = bs.findAll('p','cast_txt')# 태그, 클래스
    return {'temp' : temp[0].text , 'desc': desc[0].text }


app = Flask(__name__)


#데코레이터(@): 앞 혹은 앞뒤로 특정한 기능이 자동으로 넣어져야할 때 쓰는 방법이 데코레이터
@app.route('/', methods=['POST','GET']) 
def home():
    print("call")
    return {'fulfillmentText': 'Hello man~~'}


@app.route('/weather') 
def weather():
    city = request.args.get('city')
    info = getweather(city)
    print(info)
    #return jsonify(info) 
    #return json.dump(info) #unicode 로 변환해줌
    return info
    
    
@app.route('/dialogflow', methods=['POST','GET']) 
def dial():
    res = {'fulfillmentText': 'Hello~~'}
    #return jsonify(res)
    return res    
   
    
    
    
    
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port =80, debug=True) #debug 모드 True 면 변경사항 바뀌면 알아서 서버가 재실행 됨