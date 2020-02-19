from django.urls import path
from . import views
 
urlpatterns = [
    path('', views.index),
    path('calcform', views.calcForm),
    path('calc', views.calc),

    path('loginform', views.loginForm),
    path('login', views.login),

    path('uploadform', views.uploadForm),
    path('upload', views.upload),

    path('runpythonform', views.runpythonForm),
    path('runpython', views.runpython),

]