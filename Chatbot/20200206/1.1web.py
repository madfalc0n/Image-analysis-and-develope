from flask import Flask, request

app = Flask(__name__)


@app.route('/')
def home():
    return "<img src=/static/f14.jpg />"

  
if __name__ == '__main__':
    app.run(host='0.0.0.0', port =80, debug=True) #debug 모드 True 면 변경사항 바뀌면 알아서 서버가 재실행 됨