from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt #csrf 체크 안한다는 모듈 설정
from django.conf import settings #셋팅에서 base dir 을 사용하기 위해 호출
import module_face as face
import module_yolo as yolo
import cv2

# Create your views here.
def index(request):
    return redirect('/static/login.html')
    #return HttpResponse("Hello DJango!!!")

def test(request):
    return HttpResponse("test mode!!")
#새로운 객체가 생성되면 import 해주어야 함
#어떤 url이 들어왔을 때 실행시켜줘야 할지 설정해야 함

def test2(request):
    data = {'s':{'img':'test.jpg'}, 'list':[1,2,3,4,5] , 'message':'안녕'}  # 장고에서는 dict 타입으로 생성 ? 호출시 d['s']['img']
    return render(request, 'template.html',
    data
    ) #templates로 지정된 폴더에서 templates.html를 읽어서 호출해줌, 어디에 있는지 따로 경로 설정을 해주어야 함

def login(request):
    id = request.GET['id']
    pwd = request.GET['pwd']
    if id == pwd :
        request.session['user'] = id
        #return HttpResponse("login 성공~~  <a href=/service>서비스로 </a> ")
        return redirect('/service') # id,pwd 같으면 해당 경로로 리다이렉트, 주소창에 id,pwd 값 표시가 되지 않음
        #return service(request)
    #return HttpResponse("login 실패ㅠㅠ <a href=/static/login.html>로그인 페이지로 </a>") #브라우저 관점에서 경로를 설정해야 함
    return redirect("/static/login.html")
    

def logout(request):
    request.session['user'] = ''
    return redirect('/static/login.html')



def service(request):
    if request.session.get('user', '') == '': #만약 세션 값이 없다면
        return redirect('static/login.html') # 로그인 페이지로 리다이렉트
    html = 'main service<br> ' + request.session.get('user') + '님 감사합니다 어서오세요'
    return HttpResponse(html) #세션이 존재한다면 메인 서비스로 이동


def face_service(request):
    if request.session.get('user', '') == '': #만약 세션 값이 없다면
        return redirect('static/login.html') # 로그인 페이지로 리다이렉트
    user_n  = request.session.get('user')
    return render(request, 'face_template.html', {'user': user_n})


# @csrf_exempt
# def uploadimage(req):   
#     file = req.FILES['file1']
#     filename = file.name    
#     fp = open(settings.BASE_DIR + "/static/" + filename, "wb")
#     for chunk in file.chunks() :
#         fp.write(chunk)
#     fp.close()            
#     html =  "ok :" + "^^" + filename       
#     return HttpResponse(html)  


@csrf_exempt
def uploadimage(request):   

    file = request.FILES['file1']
    print('셀렉터 가져오기전 ')
    selector = int(request.POST["algorithm"])
    print("셀렉터까지 가져옴")
    
    filename = file._name
    print(filename)
    fp = open(settings.BASE_DIR + "/static/images/" + filename, "wb")
    for chunk in file.chunks() :
        fp.write(chunk)
    fp.close()
    
    if selector == 0:
        print("YOLO 먹힘")
        result = yolo.yolo(settings.BASE_DIR + "/static/images/" + filename)
    else:
        print("face recognition 먹힘")
        result = face.face(settings.BASE_DIR + "/static/images/" + filename)

    #print(settings.BASE_DIR + "/static/" + filename)
    #result = face.face(settings.BASE_DIR + "/static/" + filename)
    #result = face.face(filename)
    #print("result 결과: ",result)
    #cv2.imwrite(settings.BASE_DIR + "/static/" + filename,result)

    if result != "" : 
        request.session["user"] = filename    
        #return redirect("/service")
        return redirect("/face_service")
    return redirect("/static/login.html")