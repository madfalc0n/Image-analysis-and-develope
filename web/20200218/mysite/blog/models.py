from django.db import models
from django.utils import timezone


# Create your models here.

class Post(models.Model): # 기본적으로 제공된 소스
    author = models.ForeignKey('auth.User', on_delete=models.CASCADE) #auth.User는 시스템이 만든 테이블 , cascade 단계별로 이루어진 , 원래 특정 유저와 관련된거 다 지워야 함, on_delete 는 해당 유저 삭제시 같이 삭제하라는 옵션
    title = models.CharField(max_length=200)
    text = models.TextField()  # 글자수에 제한 없는 텍스트
    created_date = models.DateTimeField(
        default=timezone.now)  # 날짜와 시간
    published_date = models.DateTimeField(
        blank=True, null=True) #  필드가 폼에서 빈 채로 저장되는 것을 허용, DB 필드에 NULL 필드 허용할건지(DB관점에서) , blank는 빈값을 허용할 건지(장고 form에서 )

    def publish(self):
        self.published_date = timezone.now() 
        self.save()

    def __str__(self):
        return self.title
