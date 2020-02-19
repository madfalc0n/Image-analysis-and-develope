from django.urls import path
from . import views

 
urlpatterns = [
    #path('', views.index),
    #path('<name>/', views.index2),
    #path('<int:pk>/detail', views.index3),

    path('login/', views.LoginView.as_view(), name="login"),
    path('list/', views.list, name="list"),
    path('<int:pk>/detail/', views.detail, name='detail'),
    path('<int:pk>/edit/', views.PostEditView.as_view(), name="edit"),

]