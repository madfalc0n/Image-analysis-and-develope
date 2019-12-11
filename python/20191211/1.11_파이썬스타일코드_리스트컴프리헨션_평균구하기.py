# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:41:46 2019

@author: madfalcon

아래 코드를  *  또는  **를 사용하여 unpacking하고 zip함수를 이용해서
학생별 평균을 구하시오

map 함수와 zip함수를 사용하여 학생별 평균을 구한후에
reduce함수로 전체 평균을 구하는 코드를 작성하시오
"""

from functools import reduce


kor_score = [49, 80, 20, 100, 80]
math_score = [43, 60, 85, 30, 90]
eng_score = [49, 82, 48, 50, 100]
midterm_score = [kor_score, math_score, eng_score]
student_score = [0, 0, 0, 0, 0]

"""기존 코드
i = 0
for subject in midterm_score:
    for score in subject:
        student_score[i] += score                   # 학생마다 개별로 교과 점수를 저장
        i += 1                                      # 학생 인덱스 구분
    i = 0                                           # 과목이 바뀔 때 학생 인덱스 초기화
else:
    a, b, c, d, e = student_score                   # 학생별 점수를 언패킹
    student_average = [a/3, b/3, c/3, d/3, e/3]
    print(student_average)
"""




#zip 사용, zip은 1개 이상의 리스트값이 같은 인덱스에 있을 때 병렬로 묶는 함수
"""완료
student_score = [(a+b+c) / 3 for a, b, c in zip(kor_score, math_score, eng_score)]
print(student_score)
"""


#map,zip 사용, map은 연속 데이터를 저장하는 시퀀스형에서 요소마다 같은 기능을 적용할 때 사용한다.
"""완료
student_score = list(map( lambda x: x/3, [a+b+c for a, b, c in zip(kor_score, math_score, eng_score)] ))
print(student_score)
"""


#reduce 사용, reduce는 리스트와 같은 시퀀스 자료형에 차례대로 함수를 적용하여 모든 값을 통합하는 함수
"""완료
student_score = list(map( lambda x: x/3, [a+b+c for a, b, c in zip(kor_score, math_score, eng_score)] ))
print(student_score)
print(reduce(lambda x,y: (x+y), student_score) / len(student_score))
"""



#리스트 컴프리헨션 사용
"""완료
student_score = [(a+b+c)/3 for a, b, c in zip(kor_score, math_score, eng_score)]
print(student_score)
"""