# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:09:16 2019

@author: student
"""

# =============================================================================
# class SoccerPlayer(object):
#     def __init__(self, name, position, back_number):
#         self.name = name
#         self.position = position
#         self.back_number = back_number
#     def chang_back_number(self,new_number):
#         print("선수의 등번호를 변경한다: From {0} to {1}" .format(self.back_number,new_number))
#         self.back_number = new_number
#     def __str__(self):
#         return "HELLO, My name is {0}. I play in {1} in center." .format(self.name, self.position)
#     
# jinhyun = SoccerPlayer("jinhyun","MF",10)
# 
# print("현재 선수 등번호는: ",jinhyun.back_number)
# jinhyun.chang_back_number(5)
# print("현재 선수 등번호는: ",jinhyun.back_number)
# print(jinhyun)
# 
# =============================================================================



#상속
# =============================================================================
# class Person(object): #부모클래스 선언
#     def __init__(self,name,age,gender):
#         self.name = name
#         self.age = age
#         self.gender = gender
#         
#     def about_me(self):
#         print("부모클래스 선언부")
#         print("제 이름은 {0}이고 나이는 {1} 입니다." .format(self.name,str(self.age)))
#         
#         
# class Employee(Person):
#     def __init__(self,name,age,gender,salary,hire_date):
#         super().__init__(name,age,gender)
#         self.salary = salary
#         self.hire_date = hire_date
#         
#     def do_work(self):
#         print("열심히 일을 한다.")
#         
#     def about_me(self): #부모클래스 함수 재정의
#         super().about_me() #부모클래스 함수 사용, super()는 부모클래스를 지칭
#         print("제 급여는 {0}이고 제 입사일은 {1} 입니다." .format(self.salary, self.hire_date))
#         
# person1 = Person("사람1", 30, 'man')
# person1.about_me()
# 
# 
# emp1 = Employee("일꾼1",28,'남',300,'2019/01/01')
# emp1.about_me()
# emp1.do_work()
# =============================================================================
        
#다형성, 하나이상의 형태를 가진다는 특성, 
# 같은 이름의 메서드가 다른 기능을 할 수 있도록 하는 것

 

#가시성(캡슐화, 정보은닉)
# =============================================================================
# class Product(object):
#     pass
# 
# class Inventory(object):
#     def __init__(self):
#         self.__items = [] #'__변수명' 형태로 변수를 사용할 경우 내부에서만 사용 가능
#         
#     def add_new_item(self,product):
#         if type(product) == Product:
#             self.__items.append(product)
#             print("new item added")
#         else:
#             raise ValueError("Invalid item")
#     def get_number_of_items(self):
#         return len(self.__items)
# 
#     @property   # 숨겨진 변수를 반환(볼수있게 해준다는 설정)
#     def items(self):
#         return self.__items
# 
# my_inventory = Inventory()
# items = my_inventory.items
# items.append(Product())
# =============================================================================




class SoccerPlayer(object):
     def __init__(self, name, position, back_number):
         self.name = "1"
         self.position = "2"
         self.back_number = "3ㅏ"
     def chang_back_number(self,new_number):
         print("선수의 등번호를 변경한다: From {0} to {1}" .format(self.back_number,new_number))
         self.back_number = new_number
     def __str__(self):
         return "HELLO, My name is {0}. I play in {1} in center." .format(self.name, self.position)
     

jinhyun = SoccerPlayer("1","2",3)
 
print("현재 선수 등번호는: ",jinhyun.back_number)
jinhyun.chang_back_number(5)
print("현재 선수 등번호는: ",jinhyun.back_number)
print(jinhyun)