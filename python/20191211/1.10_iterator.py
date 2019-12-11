# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:55:42 2019

@author: student
"""
#기본
def square_numbers(nums):
    result = []
    for i in nums:
        result.append(i*i)
    return result

nums1 = square_numbers([1,2,3,4,5])
print(nums1)



#함수 내에 포인터를 가지고 있음
def square_numbers2(nums):
    for i in nums:
        yield i*i 

nums2 = square_numbers2([1,2,3,4,5])
print(nums2)
#<generator object square_numbers2 at 0x00000194F4FE88C8>
print(next(nums2)) #1
print(next(nums2)) #4
print(next(nums2)) #9
print(next(nums2)) #16
print(next(nums2)) #25
#print(next(nums2))  #StopIteration


num3 = square_numbers2([1,2,3,4,5])
for num in num3:
    print(num)

