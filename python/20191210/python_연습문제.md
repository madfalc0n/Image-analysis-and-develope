# python 문제

## 1. 파이썬 인터프리터 언어는 어떤 OS에서든 작동 가능하다. - 가능
 - 플랫폼에 독립적인 언어,
 - 객체지향 언어,
 - 동적 타이핑 언어 - 변수(객체)에 값이 할당될때 타입이 결정됨
 - 함수적 프로그래밍 언어

 
## 2. 파이썬은 C나 자바에 비해 속도가 느리다. - 맞음
 - c나 자바는 컴파일 언어이며 수행속도가 더 빠르다


## 3. 파이썬 언어의 가장 큰 장점은 기계어를 직접 다룰 수 있다는 점이다. - 틀림
 - 파이썬은 직접 다룰수 없다
 - 직접 다룰수 있는건 c언어


## 4. 파이썬은 어셈블러와 같은 기계어 변환 과정(컴파일)이 필요 없다. - 틀림
 - 컴파일 과정이든 인터프리터 과정이든 컴파일 과정은 필요하다


## 5. 파이썬은 프로그램 작동 시 별도의 번역 과정이나 소스코드의 해석 없이 CPU에 직접 전달하여 처리 가능한 인터프리터 언어이다. - 틀림
 - 인터프리터 언어는 맞다, 하지만 별도의 번역과정이나 소스코드의 해석과정이 필요하다, cpu에 직접 전달과정을 거쳐서 처리


## 6. 파이썬의 특징으로 틀린 것은? - 3번
① 플랫폼에 독립적인 언어이다.
② 해당 프로그램이 해결해야 할 문제의 구성요소를 요소별로 정의하고, 각 요소의 기능과 정보를
정의하여 요소들을 결합한 후, 프로그램을 작성하는 방식이다. - 기능별로 함수화해서 모듈화
③ 코드 작성 시 실행 순서를 중심으로 순차적으로 작성한다. 
④ 실행 시점에서 각 프로그램 변수의 타입을 결정하는 언어이다. 
⑤ 소스코드 자체가 바로 실행되는 특징이 있는 언어이다. 


## 7. 파이썬 개발 환경을 결정하는 요인이 아닌 것을 모두 고르면? - 2,5 번
① 운영체제
② 웹 브라우저
③ 파이썬 인터프리터
④ 코드 편집기
⑤ 메모장




##########################################################

## 1. 다음 코드의 실행 결과를 쓰시오.
```python
>>> a = 777  #a는 int클래스로부터 생성된 객체(인스턴스)
>>> b = 777
>>> print(a == b, a is b) #id함수를 통해 주소값 확인 가능
True False
```

## 2. 다음 중 변수를 메모리에서 삭제하기 위해 사용하는 명령어는? - 1번
1. del
2. delete
3. remove
4. pop
5. clear

## 3. 빈칸에 들어갈 각각의 코드 실행 결과를 쓰시오.
```python
>>> a = 3.5
>>> b = int(3.5) # b= 3
>>> print(a**((a // b) * 2))
12.25

>>> print(((a - b) * a) // b)
0.0

>>> b = (((a - b) * a) % b)
>>> print(b)
1.75

>>> print((a * 4) % (b * 4))
0.0
```

## 4. 다음과 같은 코드 작성 시, 실행 결과로 알맞게 짝지어진 것은? 
```python
>>> a = 10.6
>>> b = 10.5
>>> print(a * b)
111.3

>>> type(a + b)
<class 'float'>

? 111.3, <class ‘int’> ? 111.3, <class ‘str> ? 111.3, <class ‘float’>
? 105.0, <class ‘int> ? 105.0, <class ‘float’>
```

## 5. 다음과 같이 코드를 작성했을 때, 실행 결과로 알맞은 것은? 
```python
>>> a = "3.5"
>>> b = 4
>>> print(a * b)
3.53.53.53.5

1. error 
2. 3.53.53.53.5 
3. 14.0 
4. 14 
5. "14"
```

## 6. a = "3.5", b = "1.5"일 때, print(a + b)의 실행 결과는?  3.51.5
① 5 ② 3.51.5 ③ a + b ④ ab 5. 2

## 7. 다음과 같이 코드를 작성했을 때, 실행 결과로 알맞은 것은?  
```python 
>>> a = '3'
>>> b = float(a)
>>> print(b ** int(a))
27.0
① TypeError 
② '27.0' 
③ 27.0 
④ 27 
⑤ '27
```

## 8. 변수(variable)에 대한 설명으로 틀린 것은? - 3번, 메모리에 저장된다
① 프로그램에서 사용하기 위한 특정한 값을 저장하는 공간이다.
② 선언되는 순간 메모리의 특정 영역에 공간이 할당된다. #stack 방식으로 실행력을 기록하는 메모리가 있다
③ 변수에 할당된 값은 하드디스크에 저장된다.
④ A = 8은 "A는 8이다"라는 뜻이 아니다.
⑤ ‘2x + 7y’는 14라고 하면, 이 식에서 x와 y가 변수이다.

## 9. 다음과 같이 코드를 작성했을 때, 실행 결과로 알맞은 것은?  
```python 
>>> a = '20'
>>> b = '4'
>>> print(type(float(a / b))) 
#문자열에서는 *연산, + 연산만 가능하다, / 연산은 가능하지 않다
1. <class 'int'> 
2. <class 'str'> 
3. <class 'float'>  
4. 4. 3.333333333 
5. TypeError
```

## 10. 다음 코드의 예상되는 실행 결과를 쓰시오. 
(가) print("1.0" * 5) 
(나) print("1.0" + 2)
(다) print("Hanbit" + "Python")
(라) print("3.5" + "0.5")

## 11. 변수명을 지을 때 권장하는 규칙 중 틀린 것은? - 2번, 한글은 나중에 문제가 될 수 있다(다른 프로그램과의 호환여부)
① 변수명은 알파벳, 숫자, 언더스코어(_ ) 등을 사용하여 표현할 수 있다.
② 변수명은 의미 있는 단어로 쓰는 것을 권장하며, 한글도 사용할 수 있다.
③ 변수명은 대소문자가 구분된다.
④ 문법으로 사용되는 특별한 예약어는 변수명으로 쓰지 않는다.
⑤ 변수명은 “a”, “b” 등으로 사용하는 것은 권장하지 않는다.


## 12. 다음 코드의 실행 결과를 쓰시오.  
```python 
a = [0, 1, 2, 3, 4] #list는 요소 변경이 가능, 사이즈가 결정되어 있지 않다
print(a[:3], a[:-3]) #list접근할 때 index 사용, 슬라이싱을 통해서 특정 인덱스 범위만 출력 가능
[0,1,2] , [0,1]
```

## 13. 다음 코드의 실행 결과를 쓰시오. 
```python 
a = [0, 1, 2, 3, 4]
print(a[::-1])
[4, 3, 2, 1, 0]
```

## 14. 다음 코드의 실행 결과를 쓰시오.  
```python 
first = ["egg", "salad", "bread", "soup", "canafe"]
second = ["fish", "lamb", "pork", "beef", "chicken"]
third = ["apple", "banana", "orange", "grape", "mango"]
order = [first, second, third]
john = [order[0][:-2], second[1::3], third[0]]
del john[2] #[['egg', 'salad', 'bread'], ['lamb', 'chicken']]
john.extend([order[2][0:1]])
print(john) #[['egg', 'salad', 'bread'], ['lamb', 'chicken'], ['apple']]
```

## 15. 다음 코드의 실행 결과를 쓰시오.  
```python 
list_a = [3, 2, 1, 4]
list_b = list_a.sort()
print(list_a, list_b) #list_b는 정렬된 결과를 반환하는건 아님
[1, 2, 3, 4] None
```

## 16. 다음 코드의 실행 결과를 쓰시오. 
```python 
a = [5, 7, 3]
b = [3, 9, 1]
c = a + b # [5, 7, 3, 3, 9, 1]
c = c.sort() #sort는 요소를 정렬할때 반환값은 none
print(c)
None
```

## 17. 다음 코드의 실행 결과를 쓰시오.  
```python 
fruits = ['apple', 'banana', 'cherry', 'grape', 'orange', 'strawberry', 'melon']
print(fruits[-3:], fruits[1::3])
['orange', 'strawberry', 'melon'] ['banana', 'orange']
```

## 18. 다음 코드의 실행 결과를 쓰시오.  
```python 
num = [1, 2, 3, 4]
print(num * 2) #시퀀스 자료형이므로 반복
[1, 2, 3, 4, 1, 2, 3, 4]
```

## 19. 다음 코드의 실행 결과를 쓰시오. 
```python 
a = [1, 2, 3, 5]
b = ['a', 'b', 'c','d','e']
a.append('g')
b.append(6)
print('g' in b, len(b))
False 6
```

## 20. 다음과 같이 코드를 작성했을 때, 실행 결과로 알맞은 것은?  - 3번
```python 
list_a = ['Hankook', 'University', 'is', 'an', 'academic', 'institute', 'located', 'in', 'South Korea']
list_b=[ ]
for i in range(len(list_a)):
	if i % 2 != 1:
	list_b.append(list_a[i])
print(list_b)
1. None
2. Error
3. ['Hankook', 'is', 'academic', 'located', 'South Korea']
4. ['University', 'an', 'institute', 'in']
5. ['Hankook', 'University', 'is', 'an', 'academic', 'institute', 'located', 'in', 'South Korea']
```

## 21. 다음 코드를 실행한 후, 2018과 "2018"을 각각 입력했을 경우 알맞은 실행 결과끼리 묶인 것은? -3번 
```python 
admission_year = input("입학 연도를 입력하세요: ") #input은 무조건 str으로 저장
print(type(admission_year))
? <class ‘str’>, <class ‘float’>
? <class ‘int’’>, <class’str’>
? <class ‘str’>, <class ‘str’>
? <class ‘int’>, <class ‘int’>
? <class ‘float’>, <class ‘int’>
```

## 22. 다음 코드의 실행 결과를 쓰시오.  
```python 
country = ["Korea", "Japan", "China"]
capital = ["Seoul", "Tokyo", "Beijing"]
index = [1, 2, 3]
country.append(capital)
country[3][1] = index[1:]
print(country)
# ['Korea', 'Japan', 'China', ['Seoul', [2, 3], 'Beijing']]
```

## 23. 다음과 같이 코드를 작성했을 때 예측되는 실행 결과를 쓰시오.
```python 
>>> a = 1
>>> b = 1
>>> a is b
# True 
 
>>> a = 300
>>> b = 300
>>> a is b
# False
```

## 24. 다음과 같이 코드를 작성했을 때 예측되는 실행 결과를 쓰시오.
```python 
>>> a = [5, 4, 3, 2, 1]
>>> b = a
>>> c = [5, 4, 3, 2, 1]
>>> a is b
# True

>>> a is c
# False
```

## 25. 주어진 자연수 N에 대해 N이 짝수이면 N!을, 홀수이면 ΣN을 구하는 코드를 작성
```python 
n = input("자연수 N입력: ")
if n%2 == 0:
	result =1
	for i in range(1,n+1):
		result = result*i
else:
	result = 0
	for i in range(1,n+1):
		result = result+i
```

## 26. 다음 코드의 실행 결과를 쓰시오.  
```python 
fruit == 'apple'
if fruit == 'Apple':
	fruit = 'Apple'
elif fruit == 'fruit':
	fruit = 'fruit'
else:
	fruit = fruit
print(fruit)
#apple
```


## 27. 다음 코드의 실행 결과를 쓰시오.  
```python 
num = ['12', '34', '56']
for i in num:
	i = int(i)
print(num)
# ['12', '34', '56']
```


## 28. 다음 코드의 실행 결과를 쓰시오.  
```python 
number = ["1", 2, 3, float(4), str(5)]
if number[4] == 5:
	print(type(number[0]))
elif number[3] == 4:
	print(number[2:-1])
# [3, 4.0]
```

## 29. 다음 코드의 실행 결과를 쓰시오.  
```python 
num = 0
i = 1
while i < 8:
	if i % 3 == 0:
		break
	i += 1
	num += i
print(num)
# 5
```

## 30. 다음 코드의 실행 결과를 쓰시오.  
```python 
result = 0
for i in range(5, -5, -2):
	if i < -3:
	result += 1
else:
	result -= 1
print(result)
# -5
```

## 31. 다음 코드의 실행 결과를 쓰시오.
```python 
fruit = 'apple'
if fruit == 'Apple':
	fruit = 'Apple'
elif fruit == 'fruit':
	fruit = 'fruit'
else:
	fruit = fruit
print(fruit)
# apple
```

## 32. 다음 코드의 실행 결과를 쓰시오.
```python 
first_value = 0
second_value = 0
for i in range(1, 10):
	if i is 5:
		continue
		first_value = i
	if i is 10:
		break
		second_value = i
print(first_value + second_value)
# 0
```

## 33. 다음 코드의 실행 결과를 쓰시오. 
```python 
num = ""
for i in range(10):
	if i <= 5 and (i % 2)==0:
		continue
	elif i is 7 or i is 10:
		continue
	else:
		num = str(i) + num
print(num)
# 986531
```