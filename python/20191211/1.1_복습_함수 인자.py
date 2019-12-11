def spam(a,b):
    print("a 주소 ", id(a))
    print("b 주소 ", id(b))
    temp = a[0]
    a[0] = b[0]
    b[0] =temp

x = [10]
y = [100]
print("x 주소 ", id(x))
print("y 주소 ", id(y))
spam(x,y)
print(x)
print(y)


print()
def spam(c,d):
    print("c 주소 ", id(c))
    print("d 주소 ", id(d))
    temp = c
    c = d
    d =temp

x = 10
y = 100
print("x 주소 ", id(x))
print("y 주소 ", id(y))
spam(x,y)
print(x)
print(y)
