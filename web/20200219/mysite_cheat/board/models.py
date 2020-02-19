from django.db import models
from django.utils import timezone
from django.db import models


# Create your models here.

class Board(models.Model):
    author = models.ForeignKey('auth.User', on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    text = models.TextField()  # 글자수에 제한 없는 텍스트
    created_date = models.DateTimeField(
        default=timezone.now)  # 날짜와 시간
    cnt = models.IntegerField(default=0)
    category = models.CharField(max_length=20, default="common")
    image = models.CharField(max_length=50, null=True, blank=True)


    def __str__(self):
        return self.title

