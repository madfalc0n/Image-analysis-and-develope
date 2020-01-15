from flask import Flask, escape, request

app = Flask(__name__)

@app.route('/hello') #디렉토리별 지정 가능
def hello():
    print(1)
    name = request.args.get("name", "World")
    return f'Hello, {escape(name)}!'


@app.route('/hello2',methods=['GET', 'POST']) # GET POST 방식 모두 사용, default 는 GET 
def hello2():
    #name = request.args.get("name", "World")
    return {
    "version": "2.0",
    "template": {
        "outputs": [
            {
                "simpleImage": {
                    "imageUrl": "http://k.kakaocdn.net/dn/83BvP/bl20duRC1Q1/lj3JUcmrzC53YIjNDkqbWK/i_6piz1p.jpg",
                    "altText": "보물상자입니다"
                }
            }
        ]
    }
}