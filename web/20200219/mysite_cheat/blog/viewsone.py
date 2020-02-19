from django.shortcuts import render, get_object_or_404,redirect
from django.http import HttpResponse
from blog.models import Post
from django.views.generic import View
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from django.forms import Form
from django.forms import CharField, Textarea, ValidationError


def list(request) :
    username = request.session["username"]
    user = User.objects.get(username=username)
    data = Post.objects.all().filter(author=user)
    context = {"data":data, "username":username}
    return render(request, "blog/list.html", context)

def detail(request, pk) :
    p = get_object_or_404(Post, pk=pk)
    return render(request, "blog/detail.html", {"d":p})


class PostEditView(View) :
    def get(self, request, pk):
        if pk == 0 :
            form = PostForm()
        else :
            post = get_object_or_404(Post, pk=pk)
            form = PostForm(initial={'title': post.title, 'text': post.text})
        return render(request, "blog/edit.html", {"form":form})
    
    def post(self, request, pk):
        form = PostForm(request.POST)
        if form.is_valid():
            if pk == 0:
                username = request.session["username"]
                user = User.objects.get(username=username)
                Post.objects.create(title=form['title'].value(), text=form['text'].value(), author=user)
            else :
                post = get_object_or_404(Post, pk=pk)
                post.title = form['title'].value()
                post.text = form['text'].value()
                post.publish()
            return redirect("list")
        return render(request, "blog/edit.html", {"form": form})

def validator(value) :
    if len(value) < 5 : raise  ValidationError("길이가 너무 짧아요");

class PostForm(Form):
    title = CharField(label='제목', max_length=20, validators=[validator])
    text = CharField(label="내용", widget=Textarea)

class LoginView(View) :
    def get(self, request):
        return render(request, "blog/login.html")

    def post(self, request):
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(username=username, password=password)
        if user == None :
            return redirect("login")
        request.session["username"] = username
        return redirect("list")