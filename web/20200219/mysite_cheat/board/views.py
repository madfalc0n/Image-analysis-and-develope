from django.shortcuts import render, get_object_or_404,redirect
from django.http import HttpResponse, HttpResponseNotFound
from django.views.generic import View
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from django.forms import Form
from django.forms import CharField, Textarea, ValidationError
from django import forms
from . import forms
from . import models

class BoardView(View) :
    def get(self, request, pk, mode):
        if  mode == 'add' :
            form = forms.BoardForm()
        elif mode == 'list' :
            username = request.session["username"]
            user = User.objects.get(username=username)
            data = models.Board.objects.all().filter(author=user)
            context = {"data": data, "username": username}
            return render(request, "board/list.html", context)
        elif mode ==  "detail" :
            p = get_object_or_404(models.Board, pk=pk)
            return render(request, "board/detail.html", {"d": p})
        elif mode == "edit" :
            post = get_object_or_404(models.Board, pk=pk)
            form = forms.BoardForm(instance=post)
        else :
            return HttpResponseNotFound('<h1>Page not found</h1>')

        return render(request, "board/edit.html", {"form":form})

    def post(self, request, pk, mode):

        username = request.session["username"]
        user = User.objects.get(username=username)

        if pk == 0:
            form = forms.BoardForm(request.POST)
        else:
            post = get_object_or_404(models.Board, pk=pk)
            form = forms.BoardForm(request.POST, instance=post)

        if form.is_valid():
            post = form.save(commit=False)
            if pk == 0:
                post.author = user
                post.save()
            else :
                post.publish()
            return redirect("board", 0, 'list')
        return render(request, "board/edit.html", {"form": form})