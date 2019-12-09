year = int(input("태어난 연도를 입력하시오. : "))
age  = (2020 - year) +1
if 20 <= age <= 26:
    print("대학생")
elif 17 <= age < 20:
    print("고등학생")
elif 14 <= age < 17:
    print("중학생")
elif 8 <= age < 14:
    print("초등학생")
else:
    print("학생이 아닙니다.")
