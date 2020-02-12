from flask import Flask, render_template, request
import cv2
import module_yolo as yolo
import module_face as face

app = Flask(__name__)


listtitle = ['개구리1', '개구리2', '개구리3']
listimg = ['img1.jpg', 'img2.jpg', 'img3.jpg']
listdata = []
listdata = [{"id":0, "img":"img1.jpg", "title":"개구리1" },
            {"id":1, "img":"img2.jpg", "title":"개구리2" },
            {"id":2, "img":"img3.jpg", "title":"개구리3" }
            ]

#title 과 img 를 listdata에 매핑 시킴
#list(map(lambda x: listdata.append(x), [{'id':n , 'title': i, 'img': j} for n,i, j in zip(range(len(listtitle) , listtitle, listimg)]))

def goURL(msg, url) :
   html = """        
        <script>    
              alert("@msg");        
              window.location.href = "@url";
        </script>
            """
   html = html.replace("@msg", msg)
   html = html.replace("@url", url)
   return html



@app.route('/')
def home():
    return render_template('home.html', title="My YOLO Page")


@app.route('/image')
def image():
    return render_template('image.html', listdata=listdata)


@app.route('/view')
def image_view():
    id = request.args.get("id")
    print(id)
    return render_template('view.html', s=listdata[int(id)])


@app.route('/fileUpload', methods = ['POST'])
def fileUpload():
    f = request.files['file1']
    f.save("./static/" + f.filename)
    title = request.form.get("title")
    #id = len(listdata)
    
    selector = int(request.form.get("algorithm")) # 셀렉터 구문 받아옴
    if selector == 0:
        print("YOLO 먹힘")
        result_pic = yolo.yolo("./static/" + f.filename)
    else:
        print("face recognition 먹힘")
        result_pic = face.face("./static/" + f.filename)


    cv2.imwrite("./static/" + f.filename, result_pic)

    id = listdata[len(listdata) - 1]["id"] + 1

    listdata.append({"id": id, "img": f.filename, "title": title})
    print(listdata)
    html = """
    <script>
    alert("업로드 완료")
    window.location.href = '/image'
    </script>
    """
    #return f'{f.filename} - 제목 {title}: 업로드 성공! <br> <img src=/static/{f.filename}><br><td><a href="/image"> 목록으로</a> </td> '
    #return html
    return goURL("업로드가 성공했습니다.",  "/image")


@app.route('/filedel') #/delete?id=0 이런식으로 해야함
def filedel():
    idx = -1
    id = int(request.args.get("id"))
    print(id)
    for i in range(len(listdata)):
        if id == listdata[i]['id']:
            idx = i

    if idx >= 0: del listdata[idx]

    return goURL("이미지를 삭제했습니다.", '/image')




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)

