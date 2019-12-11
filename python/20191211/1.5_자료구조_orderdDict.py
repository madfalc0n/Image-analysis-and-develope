# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:43:55 2019

@author: madfalcon

ordereddict 모듈
value 값 기준으로 정렬된 dict를 나타낸다.
만약  key 값 기준으로 정렬하고 싶은경우 'sorted(d.items(), key=sort_by_key)' 코드를 사용한다.
"""
from collections import OrderedDict

#Value 기준 정렬
c = OrderedDict()
c['x'] = 100
c['y'] = 200
c['z'] = 300
c['l'] = 400

for k,v in c.items():
    print(k,v)





print()

#KEY 기준 정렬
def sort_by_key(t):
    return t[0]



d = dict()
d['x'] = 100
d['y'] = 200
d['z'] = 300
d['l'] = 400


for k,v in OrderedDict(sorted(d.items(), key=sort_by_key)).items():
    print(k,v)