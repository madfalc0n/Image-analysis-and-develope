from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse
from blog.models import Post  # models -> .models
from django.contrib.auth.models import User
from django.views.generic import View
from django.contrib.auth import authenticate
from django.forms import Form
from django.forms import CharField, Textarea

# Create your views here.
def index(request):
    return HttpResponse('ok')

def index2(request, name):
    return HttpResponse('ok 문자열' + name)

def index3(request, pk):
#    p = Post.objects.get(pk=pk)
    p = get_object_or_404(Post, pk=pk)
    #return HttpResponse('ok 정수' + str(pk)) #int타입이라 문자로 변경해주어야 함
    return HttpResponse('ok 정수' + p.title)

def list(request):
    username = request.session['username']
    #data = Post.objects.all()
    #data = Post.objects.all(author=username)
    user = User.objects.get(username=username)
    data = Post.objects.all().filter(author=user)
    context = {'data' : data , 'username':username}
    return render(request, 'blog/list.html', context)# 템플릿네임 없었음

def detail(request,pk):
    print(Post)
    p = get_object_or_404(Post, pk=pk)
    print(p)
    print(pk)
    return render(request, 'blog/detailview.html', {'data':p})

class PostView(View):  #상속 받아서 사용할 거, 객체 지향으로 해야 대형프로젝트 구축시 맞기 때문에
    def get(self, request):
        return render(request, 'blog/edit.html')

    def post(self, request):
        title = request.POST.get('title')
        text = request.POST.get('text')
        username = request.session['username']
        user = User.objects.get(username=username)
        Post.objects.create(title=title, text=text, author=user) # create 는 생성하고 자동으로 세이브 해줌
        return redirect('list')
    
class Postform(Form): # Form 추가
    title = CharField(label="제목", max_length=20)
    text = CharField(label="내용", widget=Textarea)

class Posteditview(View):
    def get(self, request,pk):
        #데이터 초기값 지정이 필요
        print("키값: ",pk)
        post = get_object_or_404(Post, pk=pk)
        form = Postform(initial={'title': post.title, 'text':post.text})
        return render(request, 'blog/edit.html', {'form':form, 'pk':pk})  # pk 추가

    def post(self, request):
        title = request.POST.get('title')
        text = request.POST.get('text')
        username = request.session['username']
        user = User.objects.get(username=username)
        Post.objects.create(title=title, text=text, author=user) # create 는 생성하고 자동으로 세이브 해줌
        return redirect('list')    


class LoginView(View):
    def get(self, request):
        return render(request, 'blog/login.html')

    def post(self, request):
        username = request.POST.get('username')
        #print(username)
        password = request.POST.get('password')
        #print(password)
        user = authenticate(username=username, password=password)
        if user == None:
            #return HttpResponse("암호 틀림")
            return redirect('login')
        request.session['username'] = username
        print(username)
        return redirect('list')
        #return HttpResponse("암호 맞음")

        #실제 로그인 처리가 들어가는 부분
        #return redirect('list')

