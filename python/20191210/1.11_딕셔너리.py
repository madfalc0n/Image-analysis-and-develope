#key는 unique해야 하며, 불변 이다 , value 는 가변(변경 가능)
dict = {'Name': 'Zara', 'Age': 7, 'Class': 'First'}
print ("dict['Name']: ", dict['Name'])
print ("dict['Age']: ", dict['Age'])


dict = {'Name': 'Zara', 'Age': 7, 'Class': 'First'}
#print ("dict['Alice']: ", dict['Alice'])  #존재하지 않는 키로 요소에 접근할 경우? 에러


dict['Age'] = 8;  #요소의 값 변경
dict['School'] = "DPS School"  #새로운 요소 추가
print ("dict['Age']: ", dict['Age'])
print ("dict['School']: ", dict['School'])


dict1 = {'Name': 'Zara', 'Age': 7, 'Class': 'First'}
del dict1['Name']
print(dict1)
dict1.clear() #dict의 모든 요소를 삭제하고 dict는 남는다
print(dict1) #에러 발생안함 빈 dict 출력
del dict1
#print(dict1) #에러가 발생한다.


dict1 = {'Name': 'Zara', 'Age': 7, 'Name': 'Manni'}
print ("dict1['Name']: ", dict1['Name']) #'Name' 키의 값이  Manni로 나옴


#dict2 = {['Name']: 'Zara', 'Age': 7} #키에 가변객체 ['Name'] 을 넣음, 에러발생
#print ("dict2['Name']: ", dict2['Name'])


dict3 = {'Name': 'Zara', 'Age': 7}
print ("Value : %s" %  dict3.items())
print ("Value : %s" %  dict3.keys())
print ("Value : %s" %  dict3.get('Age'))
print ("Value : %s" %  dict3.get('Sex', "NA")) #만약 키가 없을 때 요청시 다른값 반환



dict4 = {'Sex': 'female', 'Age': 7, 'Name': 'Zara'}
print ("Values : ",  list(dict4.values()))

dict5 = {'Name': 'Manni', 'Age': 7, 'Class': 'First'}
print ("Length : %d" % len (dict5))

dict6 = {'Name': 'Zara', 'Age': 7};
dict7 = {'Name': 'Mahnaz', 'Age': 27};
dict8 = {'Name': 'Abid', 'Age': 27};
dict9 = {'Name': 'Zara', 'Age': 7};
