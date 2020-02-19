from django.urls import path
from . import views

 
urlpatterns = [
    path('<int:pk>/<mode>/', views.BoardView.as_view(), name="board"),
]