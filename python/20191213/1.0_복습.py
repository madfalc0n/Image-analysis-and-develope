# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 09:41:49 2019

@author: student
"""
#객체지향 복습

"""
class Calculator:
    @staticmethod
    def plus(a, b):
        return a+b

if __name__=='__main__':
    print("main에서 발동되었ㄷ.")
    print(Calculator.plus(10,20))
"""




"""
class InstanceCounter:
    count = 0
    def  __init__(self):
        InstanceCounter.count +=1

    @classmethod
    def print_instance_count(cls):
        print("생성된 인스턴스 개수는 : ", cls.count)    

if __name__=='__main__':
    x = InstanceCounter()
    InstanceCounter.print_instance_count()
    
    x2 = InstanceCounter()
    x2.InstanceCounter.print_instance_count()
    
    x3 = InstanceCounter()
    x3.print_instance_count()
    
    
"""



#가시성(Private)
"""
class HasPrivate:
    def __init__(self):
        self.public = "public"
        self.__private="private"

    def print_internal(self):
        print(self.public)         #인스턴스 내부에서는 접근 가능
        print(self.__private)      #인스턴스 내부에서는 접근 가능

object1 = HasPrivate()
object1.print_internal()
print (object1.public)
print (object1.__private)   # 인스턴스 외부에서는 접근 불가
"""

"""
class HasPrivate:
    def __init__(self, input1, input2):
        self.public = input1
        self.__private=input2

    def print_internal(self):
        print(self.public)         #인스턴스 내부에서는 접근 가능
        print(self.__private)      #인스턴스 내부에서는 접근 가능
   
    @property
    def Private(self):
        return self.__private

    @private.setter                #@은닉변수.setter
    def Private(self, input_private):    
        self.__private= input_private


object1 = HasPrivate("public", "private")
object1.print_internal()
print (object1.__private)   # 인스턴스 외부에서는 접근 불가
print (object1.Private()) 
object1.Private("indirect")
print (object1.Private()) 

"""

"""
# 방법1:
class HasPrivate:
    def __init__(self, input1, input2):
        self.public = input1
        self.__private=input2

    def print_internal(self):
        print(self.public)         #인스턴스 내부에서는 접근 가능
        print(self.__private)      #인스턴스 내부에서는 접근 가능
   
    @property
    def private(self):
        return self.__private

    @private.setter                #메소드명.setter 데코레이터 선언
    def private(self, input_private):    
        self.__private= input_private


object1 = HasPrivate("public", "private")
print (object1.private) 
object1.private="indirect"
print (object1.private) 

#방법2 - 데코레이터 없이 일반함수로 정의함
class HasPrivate:
    def __init__(self, input1, input2):
        self.public = input1
        self.__private=input2

    def print_internal(self):
        print(self.public)         #인스턴스 내부에서는 접근 가능
        print(self.__private)      #인스턴스 내부에서는 접근 가능   
   
    def get_private(self):
        return self.__private

                          
    def set_private(self, input_private):    
        self.__private= input_private


object1 = HasPrivate("public", "private")
print (object1.get_private() ) 
object1.set_private ("indirect2" )
print (object1.get_private()) 
"""






#상속
"""
class A:
    def method(self):
        print("A")


class B:
    def method(self):
        print("B")


class C(A):
    def method(self):    #재정의
        print("C")

class D(B, C):
    pass

class E(C, B):
    pass

obj = D()
obj.method()    #?  

obj2 = E()
obj2.method()    #?  

"""





#예외처리
"""
print("start")

def getitem(idx):
    return list1[idx]

list1 = [ 100, 200, 300]
for i in range(len(list1)):
    try:
        getitem(i)
    except IndexError as ide:
        print(ide)
        print("abnormal event...and next")
    else:
        print("normally execute")
    finally:
        print("finally block execute")
        
print("END")    
"""

#정상경우
print("start")

def getitem(idx):
    return list1[idx]

list1 = [ 100, 200, 300]

try:
    print(getitem(0))
except IndexError as ide:
    print(ide)
    print("abnormal event...and next")
else:
    print("normally execute")
finally:
    print("finally block execute")
        
print("END") 
print()


##비정상경우
print("start")

def getitem(idx):
    return list1[idx]

list1 = [ 100, 200, 300]

try:
    print(getitem(3))
except IndexError as ide:
    print(ide)
    print("abnormal event...and next")
else:
    print("normally execute")
finally:
    print("finally block execute")
        
print("END") 
print()