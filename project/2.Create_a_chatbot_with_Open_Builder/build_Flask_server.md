# Flask(웹 서버 만들기)

https://www.palletsprojects.com/p/flask/ 참조

- 플라스크 마이크로서비스에 만드는거에 편리한 서비스
- 장고도 있다, 장고는 올인원 패키지, DB에 관련된 프레임워크도 있고 
- ORM 객체 만들었다 지웠다 하는 테이블이 동일하게 DB에서도 동일한 역활을 한다. 객체를 통해 DB를 관리



## 1. 절차 및 테스트

1. pip install flask

2. flask 입력

   ```python
   (base) C:\Users\student>flask
   Traceback (most recent call last):
     File "C:\ProgramData\Anaconda3\lib\site-packages\flask\cli.py", line 556, in list_commands
       rv.update(info.load_app().cli.list_commands(ctx))
     File "C:\ProgramData\Anaconda3\lib\site-packages\flask\cli.py", line 399, in load_app
       "Could not locate a Flask application. You did not provide "
   flask.cli.NoAppException: Could not locate a Flask application. You did not provide the "FLASK_APP" environment variable, and a "wsgi.py" or "app.py" module was not found in the current directory.
   Usage: flask [OPTIONS] COMMAND [ARGS]...
   
     A general utility script for Flask applications.
   
     Provides commands from Flask, extensions, and the application. Loads the
     application defined in the FLASK_APP environment variable, or from a
     wsgi.py file. Setting the FLASK_ENV environment variable to 'development'
     will enable debug mode.
   
       > set FLASK_APP=hello.py
       > set FLASK_ENV=development
       > flask run
   
   Options:
     --version  Show the flask version
     --help     Show this message and exit.
   
   Commands:
     routes  Show the routes for the app.
     run     Run a development server.
     shell   Run a shell in the app context.
   
   (base) C:\Users\student>
   ```

3. `.py` 형식의 파일 하나 만들고 다음의 소스코드 추가 후 저장(ex. main.py)

   ```python
   from flask import Flask, escape, request
   
   app = Flask(__name__)
   
   @app.route('/hello')
   def hello():
       name = request.args.get("name", "World")
       return f'Hello, {escape(name)}!'
   ```

4. `CMD` 또는 아나콘다 프롬프트에서 다음과 같이 입력

   1. `cd`명령어를 통해 main.py가 있는 디렉토리로 이동
   2. `set flask_APP=main.py` 입력
   3. `flask run` 입력 

   ```bash
   #먼저 main.py  가 있는 폴더로 이동후 다음과 같이 입력
   C:\Users\student\KMH\Image-analysis-and-develope\project\2.Create_a_chatbot_with_Open_Builder> set flask_APP=main.py
   C:\Users\student\KMH\Image-analysis-and-develope\project\2.Create_a_chatbot_with_Open_Builder> flask run # 또는 "python -m flask run" 으로 입력
    * Serving Flask app "main.py"
    * Environment: production
      WARNING: This is a development server. Do not use it in a production deployment.
      Use a production WSGI server instead.
    * Debug mode: off
    * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
   127.0.0.1 - - [15/Jan/2020 15:26:54] "[37mGET / HTTP/1.1[0m" 200 -
   #localhost:5000 으로 접속하면 다음의 로그가 발생
   127.0.0.1 - - [15/Jan/2020 15:26:54] "[33mGET /favicon.ico HTTP/1.1[0m" 404 -
   
   ```

   <img src="C:/Users/student/KMH/Image-analysis-and-develope/project/2.Create_a_chatbot_with_Open_Builder/images/image-20200115155228360.png" alt="image-20200115155228360" style="zoom:50%;" />

   

5. 예제 바꿔서 해보기

   ```python
   @app.route('/hello') #디렉토리별 지정 가능, GET 만 가능
   def hello():
       name = request.args.get("name", "World")
       return f'Hello, {escape(name)}!'
   
   @app.route('/hello2',methods=['GET', 'POST']) #GET POST  둘다 가능
   def hello2():
       name = request.args.get("name", "World")
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
   ```

   <img src="C:/Users/student/KMH/Image-analysis-and-develope/project/2.Create_a_chatbot_with_Open_Builder/images/image-20200115153827322.png" alt="image-20200115153827322" style="zoom:30%;" />

   > 접속시 홈페이지에서는 글자가 깨져서 보이지만 개발자도구로 보았을 떈 잘 받아옴을 확인

   

   <img src="C:/Users/student/KMH/Image-analysis-and-develope/project/2.Create_a_chatbot_with_Open_Builder/images/image-20200115153911857.png" alt="image-20200115153911857" style="zoom:30%;" /><img src="C:/Users/student/KMH/Image-analysis-and-develope/project/2.Create_a_chatbot_with_Open_Builder/images/image-20200115154031171.png" alt="image-20200115154031171" style="zoom:30%;" />

   > GET 메소드 와 POST 메소드, POST는 405를 주며 에러 발생

   

   <img src="C:/Users/student/KMH/Image-analysis-and-develope/project/2.Create_a_chatbot_with_Open_Builder/images/image-20200115153952808.png" alt="image-20200115153952808" style="zoom:30%;" /><img src="C:/Users/student/KMH/Image-analysis-and-develope/project/2.Create_a_chatbot_with_Open_Builder/images/image-20200115155131489.png" alt="image-20200115155131489" style="zoom:30%;" />

   > url 뒤에 파라미터 'hi222' 입력시 404 에러 발생, 지정된 경로만 접속 가능





## 2. POST 메소드와 json 형식으로  DB 저장 및 출력

> POSTMAN 프로그램을 통해 각 메소드 별로 요청시 처리 됨

- 소스

  ```python
  from flask import Flask, escape, request
  
  app = Flask(__name__)
  
  db = {
      "0": {
          "id": 0,
          "k222im": "my22323oung"
      },
      "1": {
          "id": 1,
          "k222im": "123123"
      }
  }
  id = 0
  @app.route('/users' , methods=['POST']) #사용자 정보를 받아옴으로써 POST 사용,  
  def create_user():
      global id #데코레이터가 함수를 감싸므로 global로 해주어야 함
      print(id)
      body = request.get_json()
      print(body)
      # todo body에 id를 넣어준다
      print("출력",id,body)
      body['id'] = id
      db[str(id)]=body
      print("db 값은: ",db)
      id += 1
      return body,id
  
  @app.route('/users/all', methods=['GET'])
  def select_all_user():
      return db
  
  @app.route('/users/<id>', methods=['GET'])
  def select_user(id):#<id>의 값을 받는다
      if id not in db:
          print(id+'찾을 수 없음')
          return {}, 404 
      return db[id]
  
  @app.route('/users/<id>', methods=['DELETE'])
  def delete_user(id): 
      if id not in db:
          print(id+'찾을 수 없음')
          return {}, 404
      del db[str(id)]
      return db
  
  @app.route('/users/<id>', methods=['PUT'])
  def update_user(id):
      if id not in db:
          print(id+'찾을 수 없음')
          return {}, 404
      body = request.get_json()
      body['id'] = id
      db[str(id)] = body   
      return db[str(id)]
  ```

  



## 3. DB관련 호출시

```python
import pickle

db = pickle.load('./db.bin')

pickle.dump(db, './db.bin')
```

