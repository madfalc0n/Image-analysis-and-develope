# A tuple is a sequence of immutable Python objects.
# tuple 생
tup1 = ('physics', 'chemistry', 1997, 2000)
tup2 = (1, 2, 3, 4, 5 )
tup3 = "a", "b", "c", "d"
print(tup3)

list1=[]  #empty list 생성?
tup1 = () #error
tup1 = (50,)   #요소 1개인 tuple 생성할 때 반드시 ','와 함께...

#리스트와 동일한 인덱싱 방식으로 요소에 접근, 슬라이싱 가능
print ("tup1[0]: ", tup1[0])
print ("tup2[1:5]: ", tup2[1:5])


tup1 = (12, 34.56)
tup2 = ('abc', 'xyz')

#tup1[0] = 100;  #요소 변경은 불변객체이므로 error

list1=[1]
list2=[2]
list1+list2
print(list1)
tup3 = tup1 + tup2  # tuple이 결합되어 새로운 tuple 객체 생성
print (tup3)
print(tup1)


tup = ('physics', 'chemistry', 1997, 2000);

print (tup)
del tup; #  튜플은 요소만 삭제할 수 없고 전체 튜플 인스턴스가 삭제됨
print ("After deleting tup : ")
#print (tup) # 정의되지 않은 변수 참조 에러 발생



print(len((1, 2, 3)))
print((1, 2, 3) + (4, 5, 6))
print(('Hi!',) * 4)
print(3 in (1, 2, 3))
for x in (1,2,3) : print (x, end = ' ')
