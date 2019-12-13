# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 09:00:09 2019

@author: student
"""

import os

qu = int(input("몇번 입력하시겠습니까?"))
         
os.system("cls")

for k in range(qu):

    check = 0

    st = str(input("문자열을 입력하시오."))
        

    if ((st[0] == "(") and (st[len(st)-1] == ")")):
        for i in range(len(st)):
            if(st.count("(") == st.count(")")):
                check = 1
            else:
                pass
        else:
            pass
    else:
        pass
    
    if check == 0:
        print("NO")
    else:
        print("YES")
    
str(input())