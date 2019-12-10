def add(x,y):
    return x+y

if __name__=='__main__' : #main이 되는 놈만 호출
    print(add(3,5))


'''
#__name__은 특별한 변수의 이름이다.
calc.py를 직접 실행시키면 __name__변수에 __main__값이 저장된다.
import 되면 __name__변수에 calc.py 값이 저장된다.
즉 다른곳에서 import 되면 실행되지 않는다.
