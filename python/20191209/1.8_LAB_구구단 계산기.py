#구구단
dan = int(input("구구단 몇 단을 계산할까? "))
print("구구단  ",dan,"단을 계산한다.")
for num in range(1,dan+1,1):
    for num2 in range(1,10,1):
        #print( num," * ",num2 ," = ",num * num2)

        #f = f'{num}X{num2}={num*num2}' #3.6버전  f-string
        #print(f, end=" ")
        
        gugu = "{0} X {1}={2:2d}" .format(num,num2,(num*num2)) # 3버전 부터 지원
        print(gugu, end = " ")
        
    print()
else:
    print("구구단 끝")


# %operator를 지원하지만 공식문서에서는 권장하지 않는다고 함
# 포맷 관련 문서: https://docs.python.org/3/library/string.html?highlight=string%20format#string.Formatter.format
