from flask import Flask, render_template, request
app = Flask(__name__)


listData = [{"id":0, "img":"book2.jpg", "title":"책데이터" },
            {"id":1, "img":"dog.jpg", "title":"개 영상 테스트" },
            {"id":2, "img":"single.jpeg", "title":"사람 이미지 테스트" }
            ]

@app.route('/')
def index():
    return render_template('home.html', title="my home page")

@app.route('/image')
def image(): 
    return render_template('image.html', listData=listData)

@app.route('/view')   # /view?id=0
def view():    
    id = request.args.get("id")
    return render_template('view.html', s = listData[int(id)] )


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

@app.route('/fileUpload', methods = ['POST'])
def fileUpload():
    f = request.files['file1']
    f.save("./static/" + f.filename)
    title = request.form.get("title")    
    
    id = listData[len(listData)-1]["id"] + 1
    listData.append({"id":id, "img":f.filename, "title":title})    
    return goURL("업로드가 성공했습니다.",  "/image")

@app.route('/deleteimage')  # /deleteage?id=0
def deleteimage():        
    idx = -1
    id = int(request.args.get("id"))   
    for i in range(len(listData)) :
        if id == listData[i]["id"] :
            idx = i            
    if idx >= 0 : del listData[idx]                
        
    return goURL("자료를 삭제했습니다.",  "/image")



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)