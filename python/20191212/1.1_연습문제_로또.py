# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:50:20 2019

@author: student
"""
#로또
import random

#print(random.randrange(1,46))

print("1부터 45 사이 임의의 수 6개를 받는다.")
lotto_list = []
i = 6
while i > 0:
    value = random.randrange(1,46)
    if value in lotto_list:
        print("값 중복됨!!!")
        continue
    else:
        lotto_list.append(value)
        i -= 1
else:
    print(lotto_list)


