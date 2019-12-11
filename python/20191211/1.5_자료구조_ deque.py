# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:38:26 2019

@author: madfalcon
deque 같은경우 연결리스트를 지원함, 데이터를 저장할 때 요소의 값을 한 쪽으로 연결후,
요소의 다음 값의 주소값을 저장하여 데이터를 연결하는 기법이다.
즉 원형으로 저장할 수 있다

"""

from collections import deque

deque_list = deque()

for i in range(5):
    deque_list.append(i)
    
print(deque_list)
deque_list.rotate(2)
print(deque_list)
deque_list.rotate(2)
print(deque_list)
