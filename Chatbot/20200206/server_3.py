from flask import Flask, request, jsonify
import requests
import urllib
import json
from bs4 import BeautifulSoup
import pickle

def getWeather(city) :    
    url = "https://search.naver.com/search.naver?query="
    url = url + urllib.parse.quote_plus(city + "날씨")
    print(url)
    bs = BeautifulSoup(urllib.request.urlopen(url).read(), "html.parser")
    temp = bs.select('span.todaytemp')    
    desc = bs.select('p.cast_txt')    
    return  {"temp":temp[0].text, "desc":desc[0].text}   

def getQuery(word) :
    url = "https://search.naver.com/search.naver?where=kdic&query="
    url = url + urllib.parse.quote_plus(word)
    print(url)
    bs = BeautifulSoup(urllib.request.urlopen(url).read(), "html.parser")
    output = bs.select('p.txt_box')
    
    return [node.text for node  in output ]
    

def processDialog(req) :
    
    answer = req['queryResult']['fulfillmentText']
    intentName = req['queryResult']['intent']['displayName'] 
    
    if intentName == 'query' :
        word = req["queryResult"]['parameters']['any'] 
        text = getQuery(word)[0]                
        res = {'fulfillmentText': text}  
        
    elif  intentName == 'order2' :
        price = {"짜장면":5000, "짬뽕":10000, "탕수육":20000}
        params = req['queryResult']['parameters']['food_number']        
        output = [  food.get("number-integer", 1)*price[food["food"]]  for food in params ] 
        res = {'fulfillmentText': sum(output)}  
    elif intentName == 'weather'  :        
        date = req['queryResult']['parameters']['date']
        geo_city = req['queryResult']['parameters']['geo-city']                                
        info = getWeather(geo_city)  
        res = {'fulfillmentText': f"{geo_city} 날씨 정보 : {info['temp']} /  {info['desc']}"}  
    else :
        res = {'fulfillmentText': answer}           
        
    return res
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route('/')
def home():    
    name = request.args.get("name")    
    item = request.args.get("item")    
    return "hello--^^^^---" + name + item


@app.route('/abc')
def abc():
    return "test~~~~~~~"

@app.route('/weather',methods=['POST', 'GET'])
def weather():
    city = request.form.get("city")
    info = getWeather(city)    
    return  jsonify(info)

@app.route('/dialogflow', methods=['POST', 'GET'])
def dialogflow():
    
    if request.method == 'GET' :
        file = request.args.get("file")        
        with open(file, encoding='UTF8') as json_file:
            req = json.load(json_file)    
            print(json.dumps(req, indent=4, ensure_ascii=False))            
    else :
        req = request.get_json(force=True)    
        print(json.dumps(req, indent=4, ensure_ascii=False))    
    
    
    return  jsonify(processDialog(req))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)