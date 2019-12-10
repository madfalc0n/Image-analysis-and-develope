'''
1. 난수 생성 - > 변수에 저장
2. 사용자 입력 - > int()
3. 저장된 난수와 입력값 비교, up 또는 down 일경우 최대 5회 까지 반복 또는 맞출경우 스탑

'''

import random
random_value = random.randint(1,100)

print("숫자를 입력하시오. (1 ~ 100) ")
count = 1
while 1:
    user_input = int(input("입력 값: "))
    if random_value == user_input:
        print("정답입니다. 입력한 숫자는 {0}입니다." .format(user_input))
        break
    elif random_value < user_input:
        print("숫자가 너무 큽니다.")
    elif random_value > user_input:
        print("숫자가 너무 작습니다.")
    count += 1
    if count == 6:
        print("정답은 {0} 입니다." .format(random_value))
        break


    
    

