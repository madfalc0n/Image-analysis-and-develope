def spam(eggs):
    print("eggs 값 ", eggs)
    eggs.append(1)
    print("eggs 주소 ", id(eggs))
    eggs = [2,3]
    print(id(eggs))
    print(eggs)
    
ham = [0]
print("스팸 주소 ", id(ham))
spam(ham)
print(ham)


'''
1. ham에서 리스트에 대한 주소를 가지고있는다.
2. spam 함수에 ham list를 보내고 eggs는 ham과 동일한 주소를 갖는다.
 - ham 주소가 0x500 이면 eggs 도 0x500
3. eggs.append(1)을 하면 0x500 주소안에 값이 추가되므로 [0,1]이 된다.
4. eggs = [2,3] 을 선언하고나서 eggs 는 새 주소를 갖는다.
5. spam 의 주소는 0x500이므로 리스트 [0,1]을 갖는다.
6. call by reference(함수에 인수를 넘길 때 메모리 주소를 넘김)가 발생한다.

```
