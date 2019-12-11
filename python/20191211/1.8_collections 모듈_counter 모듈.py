# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:30:57 2019

@author: madfalcon
"""

from collections import Counter

c = Counter(a=4, b=2, c=0, d=-2)
d = Counter(a=1, b=2, c=3, d=4)
c.subtract(d) # c-d
print(c)


c = Counter({'red':4, 'blue': 2})
print(c)
#Counter({'red': 4, 'blue': 2})
print(list(c.elements()))
#['red', 'red', 'red', 'red', 'blue', 'blue']


c = Counter(cats=4, dogs=8)
print(c)
#Counter({'dogs': 8, 'cats': 4})
print(list(c.elements()))
#['cats', 'cats', 'cats', 'cats', 'dogs', 'dogs', 'dogs', 'dogs', 'dogs', 'dogs', 'dogs', 'dogs']



c = Counter(a=4, b=2, c=0, d=-2)
d = Counter(a=1, b=2, c=3, d=4)
print(c + d) #Counter({'a': 5, 'b': 4, 'c': 3, 'd': 2}) , 둘의 합집합

print(c & d) #Counter({'b': 2, 'a': 1}) , 둘의 교집합

print(c | d) #Counter({'a': 4, 'd': 4, 'c': 3, 'b': 2}) ,둘 중 값이 큰 것



