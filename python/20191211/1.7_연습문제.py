# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 13:07:06 2019

@author: madfalcon
"""



########## 프로그래밍 연습문제  #################
"""
1. 사용자로부터 정수값을 최대 10개까지 입력 받을 수 있습니다.
   사용자가 정수값 0을 입력하면 더 이상 입력을 받지 않고
   0 이전에 입력된 정수값들중 두번째로 큰 정수를 출력합니다.
"""
########## CODE ###################
max_val = 10   

print("정수값을 입력하세요.")
set_list = set()
for i in range(max_val):
    value = int(input())
    if value == 0:
        break
    set_list.add(value)

value_list = list(set_list)
print("두 번째로 큰 값은 {0} 입니다. " .format(value_list[-2]))




"""
2. 사용자로부터 알파벳 대소문자로 구성된 단어 혹은 문장을 입력받아서 
가장 많이 사용된 알파벳이 무엇인지 알아내는 프로그램을 작성하시오
가장 많이 사용된 알파벳이 여러 개 존재할 경우 ?를 출력하시오
단, 대소문자 구분 안함 
"""
########## CODE ###################
string_value = str(input("단어 또는 문장을 입력하시오."))
#print(string_value)
lower_string = string_value.lower()
#print(lower_string)

alphabet_dict = {}
max_count = 0
for i in range(len(lower_string)): #97 부터 122 까지 소문자
    if lower_string[i] in alphabet_dict:
        continue
    alphabet_dict[lower_string[i]] = lower_string.count(lower_string[i])
    if max_count < lower_string.count(lower_string[i]):
        max_count = lower_string.count(lower_string[i])

max_alphabet = ''
max_count2 = 0
for j,k in alphabet_dict.items():
    if k == max_count:
        max_alphabet = j
        max_count2 += 1
else:
    if max_count2 > 1:
        print("?")
    else:
        print("제일 많이 쓴 알파벳은 {0} 입니다." .format(max_alphabet))



