# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:32:02 2019

@author: student
"""

###노트북 프로그램 만들기

class Notebook(object):
    def __init__(self,title):
        self.title = title
        self.page_number = 1
        self.notes = {}
        
    def add_note(self, note, page = 0):
        if 0 <= self.page_number <= 300:
            if page == 0:
                self.page_number += 1
                self.notes[self.page_number] = note
                print("페이지 추가됨")
            else:
                self.notes = {page:note}
        else : 
            print("노트수가 300개라 못만듬.")
        
    def remove_note(self, page_number):
        if page_number in self.notes.keys(): #해당 페이지 넘버가 있는경우
            return self.notes.pop(page_number)
        else: #없을 때
            print("해당 페이지는 존재하지 않음")
            
    def get_number_of_pages(self):
        print("현재 페이지 수는 {0} 입니다." .format(self.page_number))
    

class Note(object):
    def __init__(self,contents = None):
        self.contents = contents
    
    def write_content(self,contents):
        self.contents = contents

    def remove_all(self):
        self.contents = ""
        
    def __str__(self):
        if self.contents == "":
            print("삭제된 노트북")
        else:
            print("Note 클래스가 동작하였음.")
        return self.contents


my_sentence1 = """ 가나다라마바사"""
note_1 = Note(my_sentence1)


my_sentence2 = """ dkdkdkkdkdkdkapdkjkflsj"""
note_2 = Note(my_sentence2)

my_notebook_1 = Notebook("첫 노트북")
my_notebook_1.add_note(my_sentence1)
my_notebook_1.get_number_of_pages()
my_notebook_1.add_note(my_sentence2)


