from django.urls import path
from . import views 

urlpatterns = [
    path('',views.index),
    path('calcform',views.calcform),
    path('calc',views.calc),

    path('loginform',views.loginform),
    path('login',views.login),

    path('uploadform',views.uploadform),
    path('upload',views.upload),

    path('runpythonform',views.runpythonform),
    path('runpython',views.runpython),
]
