'''
- LIST
1. 하나의 변수에 여러 값을 할당하는 자료형
2. 배열이라고 부르기도 한다
3. 인덱스(리스트의 각 주소라고 보면됨) 시작값은 0부터 시작한다

'''

colors = ['red', 'blue', 'green']
print(colors[0])
print(colors[2])
print(len(colors)) #리스트 길이

colors2 = ['o','b','w']
total = colors + colors2
print(total)
print('blue' in colors2)


#인덱싱과 슬라이싱
cities = ['서울', '대구', '부산', '인천', '울산', '광주', '수원']
print(cities[0:6]) #실제 인덱스 값이 0부터 5까지 속하는 데이터를 불러온다


#리버스 인덱싱
print(cities[::2]) # 2칸 단위로
print(cities[::-1])# 역으로 슬라이싱




#append, extend, insert, remove, del
#colors.appen('white')
#colors.extend['black','puple']
#colors.insert(0,'orange')
#colors.remove('red')
#color[0] = 'orange'
#del colors[0]


#packing , unpacking
t = [1,2,3]
a,b,c = t
print(t,a,b,c)


#파이썬은 리스트를 저장할때 값 자체가 아니라, 값이 위치한 메모리 주소값을 저장한다
#id는 리스트의 주소값을 보여줌
# ==은 값을 비교하는 연산,  is는 메모리의 주소를 비교하는 연산
# -5부터 256까지는 정수값을 특정 메모리 주소에 저장함

a = 300
b = 300

print("a 와 b 비교", a is b)
print(id(a))
print(id(b))



'''
c = a
print(c is a)

a = 1
b = 1
print(a is b)
print(a == b )
'''

g = 300
print(id(g))
h = 300
print(id(h))
print(g is h) #true로 나옴


#실제 스크립트 파일 내에서는 같은 주소값을 참조하지만, 명령프롬프트에서는 false로 나옴
