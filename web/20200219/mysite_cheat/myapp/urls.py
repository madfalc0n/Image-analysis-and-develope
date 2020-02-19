from django.urls import path
from . import views
 
urlpatterns = [
    path('', views.index),      
    path('test', views.test),   
    path('login', views.login), 
    path('service', views.service),        
    path('logout', views.logout),        
    path('uploadimage', views.uploadimage),
    path('calc', views.calc),
    path('listuser', views.listUser),
    path('post/<name>/edit', views.index2, name='index2'),
]