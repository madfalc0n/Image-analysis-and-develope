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
    return {
    "version": "2.0",
    "template": {
        "outputs": [
            {
                "simpleText": {
                    "text": "간단한 텍스트 요소입니다."
                }
            }
        ]
    }
}

@app.route('/users/all', methods=['GET'])
def select_all_user():
    return db

@app.route('/users/<id>', methods=['GET'])
def select_user(id):
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