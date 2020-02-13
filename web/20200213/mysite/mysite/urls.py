"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('',include('myapp.urls')), #사용자가만든 myapp 폴더 내 urls.py를 include 하라는 의미, 내가만든 어플리케이션을 참고해서 추가시켜주라는 의미로해석하면 됨
    path('admin/', admin.site.urls), # default 설정 , admin.site.urls(기본 패키지) 를 include 하라는 의미
]
