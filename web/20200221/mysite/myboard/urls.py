from django.urls import path
from . import views # '.' current 폴더의 의미, 현재폴더에서 views.py를 import 하라는 의미

urlpatterns = [
    path('',views.list_p),
    path('<category>/<int:pk>/<mode>/',views.BoardView.as_view(), name='myboard'),
    path('ajaxdel', views.ajaxdel),
    path('ajaxget', views.ajaxget),
    path('<category>/<int:page>/',views.listsql),
    path('imglist/',views.listimg, name='imglist'),
    path('imglist/upload',views.imgupload),


]
