# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 16:19:40 2019

@author: student
"""
##cx_Oracle 모듈 설치
##pip install cx_Oracle


import cx_Oracle
import os
os.putenv('NLS_LANG', 'KOREAN_KOREA.KO16MSWIN949')

conn  =  cx_Oracle.connect("scott/oracle@127.0.0.1:1521/orcl") #DB 연결
print(conn)