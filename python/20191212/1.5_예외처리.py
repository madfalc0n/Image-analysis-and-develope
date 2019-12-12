# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:57:11 2019

@author: student
"""

for i in range(10):
    try:
        print(10 / i)
    except ZeroDivisionError as e: #0으로 나누었을 경우
        print(e)
        print("0으로 나눌수 없음")
        




for i in range(10):
    try:
        result = 10 / i 
    except ZeroDivisionError as e: #0으로 나누었을 경우
        print(e)
        print("0으로 나눌수 없음")
    else:
        print(10 / i)
        
        
        
        