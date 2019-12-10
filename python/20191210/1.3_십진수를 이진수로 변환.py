decimal = 10
result = ''
while (decimal > 0):
    remainder = decimal % 2 #나머지
    decimal = decimal // 2 # 몫
    result = str(remainder) + result
print(result)




val = 5
bi = ''
while (val > 0):
    remainder = val % 2
    val = val //2
    bi = str(remainder) + bi
print(bi)
