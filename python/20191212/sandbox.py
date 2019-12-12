# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 09:08:36 2019

@author: student
"""
#
#


import keyword
import pprint


print(object)
print(object.__str__(object)) #표현하는 문자열?을 
print(id(object))
print(object.__class__)
print(type(object))
print(object.__name__)



pprint.pprint(keyword.kwlist, width=50, compact=True)
help(print)


import  sys
print( type ( sys ) )   # 모듈 클래스
sys = 100
print( type ( sys ) )   # 정수 클래스의 인스턴스



#얕은 복사와 깊은 복사, 
# 주소값만 복사하면 얕은 복사, 
# 새로운 객체를 만들고 내부 원소들도 다른 원소로 만드는 것이 깊은복사


#얕은 복사
l = [1, 2, 3, 4]
alias = l
print ( alias is l )
print ( alias == l )
print()



#깊은복사
lc = l.copy( )
print(lc == l )
print(lc  is l )
print(lc)

print()
ld = list ( l )
print( ld == l )
print( ld is l )
print(ld)



a,b = 1,2
print(a,b)

a,b = b,a
print(a,b)



#대문자 A , 소문자a, 숫자0 아스키코드 기억하기

s = "IwanttokownPython"
l = list(map(ord, s) ) #정렬과 관련
print( l )
ls = "".join( list ( map ( chr, l ) ) )
print( ls )
print( any ( [ ] ) ) 
print( any ( [1, 0, 0 ] ) )
print( all ( [ ] ) ) # 
print( all ( [1, 2, 0 ] ) )


#정규표현식 , 기억해야함


#날짜시간, datetime 모듈 기억하기

#