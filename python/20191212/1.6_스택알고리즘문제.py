# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 18:11:59 2019

@author: student
"""

#1번
value = int(input("정수 K 입력(1 ≤ K ≤ 100,000)"))
result_list = []
for i in range(value):
    input_value = int(input("{0}번째 정수 입력: " .format(i+1)))
    if input_value != 0:
        result_list.append(input_value)
    elif input_value == 0:
        result_list.pop()
    
print(result_list)
print(sum(result_list))




#2번
def detect_ps(ps_value):
    #vps = '()'
    start_line = True
    while start_line:
        if len(ps_value) == 0:
            start_line = False
            return "YES"
        else:
            if ps_value[0] == ')':
                start_line = False
                return "NO"
            elif ps_value[0] == '(':
                for i in range(len(ps_value)-1):
                    if ps_value[i] == '(' and ps_value[i+1] == ')':
                        ps_value.pop(i)
                        ps_value.pop(i)
                        break
        
    
    
value = int(input("2에서 50 사이 정수 입력"))
ps_list= []
for i in range(value): #괄호 입력받기
    ps_value = str(input("{0}번쨰 값: " .format(i+1)))
    ps_list.append(list(ps_value))
print(ps_list)

result_list = []
for j in ps_list:
    if len(j)%2==0:#짝수면 판별해야됨
        if j[0] == ')' or j[len(j)-1] == '(':
            result_list.append("NO")
        else:
            result_list.append(detect_ps(j))
    else:
        result_list.append("NO")

for x in result_list:#결과 출력
    print(x)