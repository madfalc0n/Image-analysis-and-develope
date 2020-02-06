from flask import Flask, request

app = Flask(__name__)
cnt = 0

@app.route('/')
def home():
    global cnt
    cnt += 1
    return f"""
    <h1> Hello </h1> <br>
    {cnt}명이 방문 했습니다. <br>
    <iframe height="430" width="350" src="https://bot.dialogflow.com/madfalconweb"></iframe>    
    """


#조회수를 그림파일로 표시하기
@app.route('/counter')
def counter():
    global cnt
    cnt += 1

    #방법 1
    # nts = str(cnt)
    # print(type(nts))
    # img_s = ''
    # for i in nts:
    #     img_s +=  "<img src=/static/"+i+".png />"
    # return    img_s  +"명이 방문 했습니다."

    #방법2
    html = "".join([ f"<img src=/static/{i}.png />" for i in str(cnt)])
    html += "명이 방문했습니다."
    return html



if __name__ == '__main__':
    app.run(host='0.0.0.0', port =80, debug=True) #debug 모드 True 면 변경사항 바뀌면 알아서 서버가 재실행 됨