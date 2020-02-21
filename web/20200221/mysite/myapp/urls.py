from django.urls import path
from . import views # '.' current 폴더의 의미, 현재폴더에서 views.py를 import 하라는 의미

urlpatterns = [
    path('',views.index), #views.py 에 있는 index라는 함수를 실행시켜 달라는 의미
    path('test', views.test),
    #path('ajax', views.test),
    path('test/main', views.test),
    path('test2', views.test2),
    path('login',views.login),
    path('service',views.service),
    #path('uploadimage',views.uploadimage),
    #path('face_service',views.face_service),
    path('listuser',views.listuser),
]
