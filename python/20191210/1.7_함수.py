def calc(a,b,op):
    result = 0
    if op == '*':
        result = a * b
        print(result)
    
    elif op == '+':
        result = a + b
        print(result)

    elif op == '-':
        result = a - b
        print(result)

    elif op == '/':
        result = a / b
        print(result)

    else:
        result = None

    return result


print(calc(10,2,'+'))
print(calc(10,2,'-'))
print(calc(10,2,'*'))
print(calc(10,2,'/'))
