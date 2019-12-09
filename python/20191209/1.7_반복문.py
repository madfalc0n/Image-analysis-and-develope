
for looper in [1,2,3,4,5]:
    print("helllo")

print()
print("홀수------------------------")
for looper in range(1,11,2): #range(시작번호, 마지막번호, 증가값)
    print(looper)

print()
print("짝수------------------------")
for looper in range(2,11,2): #range(시작번호, 마지막번호, 증가값)
    print(looper)


print()
print("while 문, break, continue------------------------")
i = 1
while i < 10 :
    print(i)
    if i == 5:
        print("컨티뉴")
        i += 1
        continue
    if i == 8:
        print("브레이크")
        break
    i += 1

#반복문에서 break와 continue 를 통해 반복을 종료하거나 다음 회차로 넘어갈 수 있다.


print()
print("for 문, else 사용------------------------")
for i in range(10):
    print(i)
else: # 반복문 끝나고 한번 더 실행해주는 역할을 해줌
    print("끝")


print()
print("for 문, else 사용, break------else문 적용안됨-----------")
for i in range(10):
    if i ==5:
        break    
    print(i)

else: # 반복문 끝나고 한번 더 실행해주는 역할을 해줌
    print("End of Loop")



print()
print("for 문, else 사용, continue------else문 적용됨-----------")
for i in range(10):
    if i == 5:
        continue
    print(i)
else: # 반복문 끝나고 한번 더 실행해주는 역할을 해줌
    print("End of Loop")



#질문1. while 반복문에서 else는 함께 사용가능할까? -> 가능
#만약 가능하면, for문에서의 결과와 동일하게 처리되나? -> ㅇㅇ
print()
print("while 문, else 사용,break------else문 적용안됨-----------")
i = 0
while i <10:
    if i == 5:
        break
    print(i)
    i += 1
else:
    print("End of Loop")



print()
print("while 문, else 사용, continue------else문 적용됨-----------")
i = 0
while i <10:
    if i == 5:
        i += 1
        continue
    print(i)
    i += 1
else:
    print("End of Loop")









