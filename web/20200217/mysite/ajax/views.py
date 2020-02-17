from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.conf import settings

#파이썬 컴파일하기위한 모듈
import sys
from io import StringIO 


# Create your views here.
def index(request):
    
    return HttpResponse("Hello DJango ajax!!!")

def calcform(request):
    return render(request, 'ajax/calc.html') #templates - ajax 폴더 내 calc.html 을 읽어오겠다는 얘기

def calc(request):
    op1 = int(request.GET['op1'])
    op2 = int(request.GET['op2'])
    result = op1 + op2
    #return HttpResponse(f'result = {result}')
    #return HttpResponse("{'result':" + str(result) + "}") #되도록이면 json 형태로 보내라
    return JsonResponse({'error':0, 'result': result})# 딕셔너리 객체를 넣어주면 됨, json 포멧으로 바꿔줌


def loginform(request):
    return render(request, 'ajax/login.html')
def login(request):
    id = request.GET['id']
    pwd = request.GET['pwd']
    if id == pwd:
        request.session['user'] = id
        return JsonResponse({'error':0})
    else:
        return JsonResponse({'error':1, 'message':'id/pwd를 다시 확인해 주시오. '+str(id)+'과'+str(pwd)+'잘못 입력 받았어요'})


def uploadform(request) :
    return render(request, "ajax/upload.html")
#@csrf_exempt# csrf 는 해제해 주세요 안전하니깐
def upload(request) :
    file = request.FILES['file1']
    filename = file._name
    fp = open(settings.BASE_DIR + "/static/" + filename, "wb")
    for chunk in file.chunks() :
        fp.write(chunk)
    fp.close()
    fp.save(settings.BASE_DIR + "/static/save_" + filename)
    return HttpResponse("upload~")


def runpythonform(request): #form 으로 렌더링 시켜줌
    return render(request, 'ajax/runpython.html')


global_v = {}
local_v = {}
def runpython(request): #form으로 부터 받아온 데이터(code)를 code 변수에 저장하고 출력
    code = request.GET['code']

    original_stdout = sys.stdout 
    sys.stdout = StringIO()
    #exec(code) 
    exec(code, global_v,local_v)#파이썬 코드를 실행시켜주는 인터프리터   exec(    ,global, local) 글로벌과 지역변수를 저장할 파라미터, 내용이 지워져도 변수값들을 저장하고 있음
    contents = sys.stdout.getvalue()
    sys.stdout = original_stdout
    #contents = contents.replace('\n','<br>')
    return HttpResponse(contents)