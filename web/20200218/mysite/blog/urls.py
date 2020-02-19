from django.urls import path
from . import views # '.' current 폴더의 의미, 현재폴더에서 views.py를 import 하라는 의미

urlpatterns = [
    path('',views.index), #views.py 에 있는 index라는 함수를 실행시켜 달라는 의미
    #path('<name>/', views.index2),     # name이라는 파라미터에 고정이 아닌 수시로 바뀌는 동적으로 매핑,  임의의 문자
    #path('<int:pk>/detail', views.index3), # 숫자만 올수 있음
    path('list', views.list, name='list'),
    
    #path('list2', views.PostView.as_view()),
    #path('login', views.LoginView.as_view(), name='login'), #이름을 정의하면서 상대경로를 지정할 수 있게 됨
    #path('add',views.PostView.as_view(), name='add'),
    
    
    #path('<int:pk>/detail', views.detail, name='detail'),
    #path('<int:pk>/edit',views.Posteditview.as_view(), name='edit'),
    path('<int:pk>/<mode>',views.Posteditview.as_view(), name='edit'),
]
