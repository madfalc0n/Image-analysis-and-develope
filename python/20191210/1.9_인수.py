





#가변 인수, 마지막에 정의를 해야함, 
def asterisk_test(a,b,*args): #args에서 받을때 packing으로 받음
    return a+b+sum(args) #sum에서 unpacking이 됨

print(asterisk_test(1,2))
print(asterisk_test(1,2,3))
print(asterisk_test(1,2,3,4))
print(asterisk_test(1,2,3,4,5))
#print(asterisk_test(1,2,(3,4,5))) #에






#키워드 가변 인수
def kwargs_test(a, b, **args): 
    return a+b


print(kwargs_test(1,2,one = 3,two = 4,three = 5))

'''
def kwargs_test2(**args,a,b): 
    return a+b+sum(args)


print(kwargs_test2(one = 1,one = 2,3,4,5))
'''
