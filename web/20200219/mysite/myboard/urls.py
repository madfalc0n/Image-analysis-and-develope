from django.urls import path
from . import views # '.' current 폴더의 의미, 현재폴더에서 views.py를 import 하라는 의미

urlpatterns = [
    path('<category>/<int:pk>/<mode>/',views.BoardView.as_view(), name='myboard'),
]
