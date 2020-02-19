from django.db import models
from django.utils import timezone
from django.db import models

class User(models.Model) :
    userid = models.CharField(max_length=10, primary_key=True)
    name = models.CharField(max_length=10)
    age = models.IntegerField()
    hobby = models.CharField(max_length=20)

    def __str__(self):
        #return self.userid  + "/" + self.name + "/" + self.age
        return f"{self.userid} / {self.name} / {self.age}"


class User2(models.Model) :
    userid = models.CharField(max_length=10, primary_key=True)
    name = models.CharField(max_length=10)
    age = models.IntegerField()
    hobby = models.CharField(max_length=20)

    def __str__(self):
        #return self.userid  + "/" + self.name + "/" + self.age
        return f"{self.userid} / {self.name} -----"



class Question(models.Model):
    question_text = models.CharField(max_length=200)
    pub_date = models.DateTimeField('date published')


class Choice(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    choice_text = models.CharField(max_length=200)
    votes = models.IntegerField(default=0)

class Order(models.Model):
    user = models.ForeignKey(User,
        on_delete=models.CASCADE, )
    desc = models.CharField(max_length=100)
    order_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return '{}-{}-{}'.format(self.user.name, self.desc, self.order_date) # 외래키 출력은?

class Product(models.Model):
    order = models.ForeignKey(Order, on_delete=models.CASCADE, )
    name = models.CharField(max_length=255)
    price = models.IntegerField()
    quantity = models.IntegerField(default=0)

    def __str__(self):
        return '{}:{} / {}'.format(self.order, self.name, self.price)