#구구단
dan = int(input("구구단 몇 단을 계산할까? "))
print("구구단  ",dan,"단을 계산한다.")
for num in range(1,dan+1,1):
    for num2 in range(1,10,1):
        print( num," * ",num2 ," = ",num * num2)
else:
    print("구구단 끝")
