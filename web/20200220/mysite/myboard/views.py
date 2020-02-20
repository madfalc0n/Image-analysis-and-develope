from django.shortcuts import render, get_object_or_404,redirect
from django.http import HttpResponse, JsonResponse
from django.views.generic import View
from django.contrib.auth.models import User
from django.forms import CharField, Textarea, ValidationError

from . import forms
from . import models


#페이지 네비게이션
from django.core.paginator import Paginator


#db 대신 간단하게 만듬

def list_p(request):
    datas = [{'id':1 , 'name': '홍길동1'},
    {'id':2 , 'name': '홍길동2'},
    {'id':3 , 'name': '홍길동3'},
    {'id':4 , 'name': '홍길동4'},
    {'id':5 , 'name': '홍길동5'},
    {'id':6 , 'name': '홍길동6'},
    {'id':7 , 'name': '홍길동7'},
    ]
    
    page = request.GET.get('page',1)
    p = Paginator(datas, 3) #한페이지당 보여줄 갯수
    subs = p.page(page) #페이지 번호, 전체 페이지 등에 대한 정보를 넘겨줌
    print("call me")
    return render(request, 'myboard/page.html' , {'datas': subs , 'pages': p} )


def ajaxdel(request):
    # GET POST 구현
    pk = request.GET.get('pk')

    board = models.Board.objects.get(pk=pk)
    board.delete()

    return JsonResponse({'error':'0'})

def ajaxget(request): #페이지 번호 불러오는 곳
    datas = models.Board.objects.all().filter( category='common')
    
    page = request.GET.get('page',1) #명시 안되어 있으면 1로 저장
    p = int(page) # 웹에서 받는 데이터는 기본적으로 문자열 이므로
    
    subs = datas[(p-1)*3:(p)*3]  # 페이지당 3개 씩 출력
    datas = {'datas': [{'pk':data.pk , 'title':data.title, 'cnt':data.cnt} for data in subs ]}

    return JsonResponse(datas)

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
            p.cnt += 1
            p.save()
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
