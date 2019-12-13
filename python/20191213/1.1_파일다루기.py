# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 11:25:11 2019

@author: student
"""
"""
f = open("today.txt","r") #경로  및 인코딩 맞추는게 중요함
read_f = f.read()

print(type(read_f))

f.close()
"""


#with를 통한 파일 읽어오기
"""
with open("today.txt","r") as my_file:
    contents = my_file.read()
    word_list = contents.split(" ")
    line_list = contents.split("\n")
    
print("총 글자의 수 : {}" .format(len(contents)))
print("총 단어의 수 : {}" .format(len(word_list)))
print("총 줄의 수 : {}" .format(len(line_list)))
"""


#파일 쓰기
"""
f = open("count_log.txt",'w', encoding = 'utf8')
for i in range(1,11):
    data = ("{0} 번째 줄\n" .format(i))
    f.write(data)
    
f.close
"""
"""
#기존 파일에 추가하기
with open("count_log.txt", 'a', encoding='utf8') as f:
    for i in range(1,11):
        data = ("{0} 번째 줄\n" .format(i))
        f.write(data)
"""

"""
import os
#print(dir(os))

#print(os.path.isdir("log"))
#os.mkdir("log") 
#os.access # 특정위치의 경로에 접근 가능한지 체크
#os.chdir("log") # 현재 디렉토리의 위치를 변경
#os.getcwd() #현재 작업디렉토리 확인
#os.listdir("path") #인수path 경로 아래파일, 디렉토리 리스트 변환
#os.path.exists("path") #파일 또는 디렉토리 경로가 존재하는지 체크
#os.path.isDir()
#os.path.isFile()
#os.path.getsize('file path')
#os.rmdir("dir name") 
#os.rename(old, new) #파일이름변경
#os.system("cls") # 시스템운영체제 명령어, 프로그램을 호출
#os.unlink("file path") #파일삭제
#os.stat #파일 정보를 반환

print(os.getcwd())
os.chdir("c:/Temp")
print(os.getcwd())
os.chdir("C:/Users/student/KMH/Image-analysis-and-develope/python/20191213")
os.listdir("C:/Users/student/KMH/Image-analysis-and-develope/python/20191213")

"""



import pickle
"""
f = open("list.pickle","wb") #바이너리 파일로 쓰기
test = [1,2,3,4,5]
pickle.dump(test,f)
f.close()
"""
"""
f = open("list.pickle","rb") #바이너리 파일 읽어오기
test_pickle = pickle.load(f)
print(test_pickle)
f.close()
"""






