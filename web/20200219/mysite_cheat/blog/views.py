from django.shortcuts import render, get_object_or_404,redirect
from django.http import HttpResponse
from django.views.generic import View
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from django.forms import Form
from django.forms import CharField, Textarea, ValidationError
from django import forms
#from blog.forms import PostForm
from . import forms
from . import models

class PostEditView(View) :
    def get(self, request, pk, mode):
        if  mode == 'add' :
            form = forms.PostForm()
        elif mode == 'list' :
            username = request.session["username"]
            user = User.objects.get(username=username)
            data = models.Post.objects.all().filter(author=user)
            context = {"data": data, "username": username}
            return render(request, "blog/list.html", context)
        elif mode ==  "detail" :
            p = get_object_or_404(models.Post, pk=pk)
            return render(request, "blog/detail.html", {"d": p})
        elif mode == "edit" :
            post = get_object_or_404(models.Post, pk=pk)
            form = forms.PostForm(instance=post)
        else :
            return HttpResponse("error page")

        return render(request, "blog/edit.html", {"form":form})

    def post(self, request, pk, mode):

        username = request.session["username"]
        user = User.objects.get(username=username)

        if pk == 0:
            form = forms.PostForm(request.POST)
        else:
            post = get_object_or_404(models.Post, pk=pk)
            form = forms.PostForm(request.POST, instance=post)

        if form.is_valid():
            post = form.save(commit=False)
            if pk == 0:
                post.author = user
                post.save()
            else :
                post.publish()
            return redirect("edit", 0, 'list')
        return render(request, "blog/edit.html", {"form": form})


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