from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import face
from django.http import JsonResponse
import json
from myapp.models import User2

def index(request):
    return HttpResponse("ok")

def test(request):
    data = {"s":{"img":"test.png"}, 
        "list":[1,2,3,4,5]  }   
    return render(request, 'myapp/template.html', data) 
    
def login(request):
    id = request.GET["id"]    
    pwd = request.GET["pwd"]
    if id == pwd :         
        request.session["user"] = id
        return redirect("/service")
    return redirect("/static/login.html")

def logout(request):
    request.session["user"] = ""
    #request.session.pop("user")
    return redirect("/static/login.html")
    
    
def service(req):  
    if  req.session.get("user", "") == "" :
        return redirect("/static/login.html") 
    html = "Main Service<br>"  + req.session.get("user") + "감사합니다<a href=/logout>logout</a>"
    return HttpResponse(html)

@csrf_exempt
def uploadimage(request):   

    file = request.FILES['file1']
    filename = file._name    
    fp = open(settings.BASE_DIR + "/static/" + filename, "wb")
    for chunk in file.chunks() :
        fp.write(chunk)
    fp.close()
    
    result = face.facerecognition(settings.BASE_DIR + "/known.bin", settings.BASE_DIR + "/static/" + filename)
    print(result)
    if result != "" : 
        request.session["user"] = result[0]    
        return redirect("/service")
    return redirect("/static/login.html")

def calc(request):
    op1 = request.GET['op1']
    op2 = request.GET['op2']
    result = int(op1) + int(op2)

    return HttpResponse( json.dumps({'result': result}), content_type='application/json')
    #return JsonResponse({'result': result})

def listUser(request) :
    if request.method == "GET" :
        userid = request.GET.get("userid", "")
        if userid != "" :
            User2.objects.all().get(userid=userid).delete()

            #User2.objects.all().filter(userid=userid)[0].delete()

            return redirect("/listuser")
        q = request.GET.get("q", "")
        data = User2.objects.all()
        if q != "" :
            data = data.filter(name__contains=q)
        return render(request, 'template2.html', {"data": data})
    else :
        userid = request.POST["userid"]
        name  = request.POST["name"]
        age = request.POST["age"]
        hobby = request.POST["hobby"]

        #User2(userid=userid, name=name, age=age, hobby=hobby).save()
        User2.objects.create(userid=userid, name=name, age=age, hobby=hobby)


        return redirect("/listuser")


def index2(request, name) :
    return render(request, str(name) + ".html")
    #return HttpResponse(f"pk={pk}")