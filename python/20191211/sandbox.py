# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 17:39:21 2019

@author: madfalcon
"""
#1번
# =============================================================================
# def work_status(task,worker,day): 
#     rest_task = task 
#     for k in range(day): 
#         if rest_task > 0:
#             rest_task = rest_task-worker
#         elif rest_task <= 0: 
#             print("Task end") 
# 
#         if rest_task > 0: 
#             print("Hire more workers")
# 
# work_status(100, 11, 10)
# print()
# work_status(100, 1, 10)
# print()
# work_status(100, 9, 10)
# print()
# work_status(100, 10, 10)
# =============================================================================



#2번
# =============================================================================
# score_list = [5, 10, 15, 20, 25, 30]
# sum_of_score = 0
# i = 0
# while i < len(score_list): 
#     if i % 2 == 0: 
#         sum_of_score += score_list[i] 
#     i += 1
# print(sum_of_score)
# =============================================================================


#3번
# =============================================================================
# coupon = 0
# money = 200000
# coffee = 3500
# while money > coffee: 
#     if coupon < 4: 
#         money = money - coffee # 196500 1  193000 2 189500 3 186000 4  
#         coupon += 1
#     else: 
#         money += 2800
#         coupon = 0
# print(money)
# =============================================================================


#4번
# =============================================================================
# list_data_a = [1, 2]
# list_data_b = [3, 4]
# for i in list_data_a: 
#     for j in list_data_b: 
#         result = i + j
# print(result)
# =============================================================================


#5번
# =============================================================================
# list_1 = [[1, 2], [3], [4, 5, 6]]
# a,b,c = list_1
# list_2 = a + b + c #1,2,3,4,5,6
# print(list_2)
# =============================================================================

#6번
# =============================================================================
# def test(t): 
#     t = 20
#     print ("In Function:", t) 
# 
# x = 10
# print ("Before:", x)
# test(x)
# print ("After:", x)
# =============================================================================


#7번
# =============================================================================
# def sotring_function(list_value): 
#     return list_value.sort()
# 
# print(sotring_function([5,4,3,2,1]))
# =============================================================================



#8번
# =============================================================================
# number = "100"
# def midterm(number): 
#     result = ""
#     if number.isdigit() is True: 
#         if number is 100: 
#             if number/10 == 1: 
#                 result = True
#             else: 
#                 result = False
#     return result
# =============================================================================

#9번
# =============================================================================
# def is_yes(your_answer): 
#     if your_answer.upper() == "YES" or your_answer.upper() == "Y": 
#         result = your_answer.lower()
# print(is_yes("Yes"))
# =============================================================================

#10번
# =============================================================================
# def add_and_mul(a, b, c): 
#     return b + a * c + b
# print(add_and_mul(3, 4, 5) == 63)
# =============================================================================

#11
# =============================================================================
# def args_test_3(one, two, *args, three): 
#     print(one + two + sum(args)) 
#     print(args)
# 
# args_test_3(3, 4, 5, 6, 7)
# =============================================================================

#12
# =============================================================================
# def rain(colors): 
#     colors.append("purple") 
#     colors = ["green", "blue"] 
#     return colors
# 
# rainbow = ["red", "orange"]
# print(rain(rainbow))
# =============================================================================

#13
# =============================================================================
# def function(value): 
#     print(value ** 3)
# 
# print(function(2))
# =============================================================================

#14
# =============================================================================
# def get_apple(fruit): 
#     fruit = list(fruit) 
#     fruit.append("e") 
#     fruit = ["apple"] 
#     return fruit
# 
# fruit = "appl"
# get_apple(fruit)
# print(fruit)
# =============================================================================

#15, 재귀
# =============================================================================
# def return_sentence(sentence, n): 
#     sentence += str(n) 
#     n -= 1
#     if n < 0: 
#         return sentence
#     else: 
#         return(return_sentence(sentence, n))
# 
# sentence = "I Love You"
# print(return_sentence(sentence, 5))
# =============================================================================

#16
# =============================================================================
# def test(x, y): 
#     tmp = x
#     x = y
#     y = tmp
#     return y.append(x) 
# 
# x = ["y"]
# y = ["x"]
# test(x, y)
# print(y)    
# =============================================================================

#17
# =============================================================================
# def countdown(n): 
#     if n %2 == 0: 
#          print ("Even") 
#     else: 
#          print ("Odd") 
#          countdown(n-1)
# countdown(3)
# =============================================================================

#18
# =============================================================================
# def calculrate_rectangle_area(rectangle_x,rectangle_y):    
#     rectangle_x = 3
#     rectangle_y = 5 
#     result = rectangle_x * rectangle_y    
#     return result    
# 
# rectangle_x = 2 
# rectangle_y = 4
# =============================================================================

#19
# =============================================================================
# def exam_func():    
#     x = 10    
#     print("Value:", x)
# x = 20 
# exam_func() 
# print("Value:", x)
# =============================================================================


#20
# =============================================================================
# a = 11 
# b = 9
# print('a' + 'b') 
# =============================================================================

#21
# =============================================================================
# fact = "Python is funny" 
# print(str(fact.count('n') + fact.find('n') + fact.rfind('n')))
# =============================================================================

#22
# =============================================================================
# text = 'Gachon CS50 - programming with python'
# text2 = " Human cs50 knowledge belongs to the world "
# text.lower()
# print(text[:5] + text[-1] + text[6] + text2.split()[0])
# =============================================================================

#23
# =============================================================================
# class_name = 'introduction programming with python'
# for i in class_name:   
#     if i == 'python':        
#         i = i.upper()       
# print(class_name)
# =============================================================================

#24
# =============================================================================
# a = '10' 
# b = '5-2'.split('-')[1]
# print(a * 3 + b)
# =============================================================================

#25
# =============================================================================
# name = "Hanbit" 
# a = name.find("H") 
# b = name.count("H") * 4 
# c = len(name) * 2 + 4 
# print("REMEMBER" , str(a) + str(b) + str(c))
# =============================================================================

#26
# =============================================================================
# a = "abcd e f g" 
# b = a.split() #abcd,e,f,g
# print(b)
# c = (a[:3][0])
# print(c) 
# d = (b[:3][0][0])
# print(d)
# print(c + d)
# =============================================================================

#27
# =============================================================================
# result = "CODE2018" 
# print("{0},{1}".format(result[-1], result[-2]))
# =============================================================================


#29
# =============================================================================
# star = 1
# space = 4
# space_val = ' '
# string_val = '*'
# 
# for i in range(6):
#     if i == 0:
#         print(space_val * 5 + string_val + space_val * 5)
#     else:
#         sum_val = space_val * space + string_val * star 
#         print(sum_val + string_val + sum_val[::-1])
#         star += 1
#         space -= 1
# =============================================================================
    


#1일때 별 1개




#30




