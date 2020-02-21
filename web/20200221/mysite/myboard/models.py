from django.db import models
from django.utils import timezone


# Create your models here.

class Board(models.Model): # 기본적으로 제공된 소스
    author = models.ForeignKey('auth.User', on_delete=models.CASCADE) #auth.User는 시스템이 만든 테이블 , cascade 단계별로 이루어진 , 원래 특정 유저와 관련된거 다 지워야 함, on_delete 는 해당 유저 삭제시 같이 삭제하라는 옵션
    title = models.CharField(max_length=200)
    text = models.TextField()  # 글자수에 제한 없는 텍스트
    created_date = models.DateTimeField(
        default=timezone.now)  # 날짜와 시간
    cnt = models.IntegerField(default=0)
    image = models.CharField(max_length=200, null=True, blank=True)
    category = models.CharField(max_length=10, default='common')


    def __str__(self):
        return self.title