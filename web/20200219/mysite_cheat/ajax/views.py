from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import sys
from io import StringIO

def index(request) :
    return HttpResponse("Hello ajax~~~")

def calcForm(request) :
    return render(request, "ajax/calc.html")

def calc(request) :
    op1 = int(request.GET["op1"])
    op2 = int(request.GET["op2"])
    result = op1 + op2
    return JsonResponse({'error':0, 'result':result})

def loginForm(request) :
    return render(request, "ajax/login.html")

def login(request):
    id = request.GET["id"]
    pwd = request.GET["pwd"]
    if id == pwd :
        request.session["user"] = id
        return JsonResponse({'error': 0})
    return JsonResponse({'error': -1, 'message': 'id/pwd를 확인해주세요'})


def uploadForm(request) :
    return render(request, "ajax/upload.html")

def upload(request) :
    file = request.FILES['file1']
    filename = file._name
    fp = open(settings.BASE_DIR + "/static/" + filename, "wb")
    for chunk in file.chunks() :
        fp.write(chunk)
    fp.close()
    return HttpResponse("upload~")

def runpythonForm(request) :
    return render(request, "ajax/runpython.html")

glo = {}
loc = {}

def runpython(request) :
    code = request.GET["code"]

    original_stdout = sys.stdout
    sys.stdout = StringIO()
    exec(code, glo, loc)
    contents = sys.stdout.getvalue()
    sys.stdout = original_stdout
    contents = contents.replace("\n", "<br>")

    contents = "<font color=red>result</font><br>" + contents
    return HttpResponse(contents)