# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:15:02 2019

@author: madfalcon
"""

a = [1,2,3,4,5]
a.append(10)
print(a) #[1, 2, 3, 4, 5, 10]

a.append(20)
print(a) #[1, 2, 3, 4, 5, 10, 20]

a.pop(0)
print(a) #[2, 3, 4, 5, 10, 20]


world = input("input a word :")
world_list = list(world)
print(world)

result = []
for _ in range(len(world_list)): # for 뒤에 '_' 는 순서대로 받지 않는다?
    result.append(world_list.pop(0))
    
print(result)
print(world[::-1])