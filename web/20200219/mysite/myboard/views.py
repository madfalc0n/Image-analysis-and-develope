from django.shortcuts import render, get_object_or_404,redirect
from django.http import HttpResponse
from django.views.generic import View
from django.contrib.auth.models import User
from django.forms import CharField, Textarea, ValidationError
from . import forms
from . import models


class BoardView(View) :
    def get(self, request, pk, mode, category):
        print("보드 뷰의 현장")
        print(pk , mode, category)
        username = '이순신'
        if mode =='add': # 새 데이터 추가할 경우 빈폼으로
            form = forms.BoardForm()
        
        elif mode == 'list': 
            #request.session["username"]
            user = User.objects.get(username=username)
            if category == 'category':
                data = models.Board.objects.all().filter(author=user)
            else:
                data = models.Board.objects.all().filter(author=user, category=category)
            context = {"data":data, "username":username}
            return render(request, "myboard/list.html", context)
        
        elif mode == 'detail':
            p = get_object_or_404(models.Board, pk=pk)
            return render(request, "myboard/detailview.html", {"data":p})
        
        elif mode == 'edit' :
            post = get_object_or_404(models.Board, pk=pk)
            form = forms.BoardForm(instance=post)
        
        elif mode == 'delete':
            board = get_object_or_404(models.Board, pk=pk)
            board.delete()
            return redirect('myboard', category, 0 , 'list') # /myboard/category/0/list 로 리다이렉트

        else:
            return HttpResponse("error")
        
        return render(request, "myboard/edit.html", {"form":form})

    def post(self, request, pk, mode, category):

        #username = request.session["username"]
        username = '이순신'
        user = User.objects.get(username=username)

        if pk == 0: #  신규 데이터 추가할 항목
            form = forms.BoardForm(request.POST)
            print(form)
        else: # 기존 데이터 수정
            post = get_object_or_404(models.Board , pk=pk)
            form = forms.BoardForm(request.POST, instance=post)

        if form.is_valid(): # 폼 작성 시 유효하지 않으면
            post = form.save(commit=False)
            if pk == 0: #새로 데이터를 추가하는 케이스
                post.author = user
            post.save()
            return redirect("myboard",category,0, 'list') 
        return render(request, "myboard/edit.html", {"form": form})
