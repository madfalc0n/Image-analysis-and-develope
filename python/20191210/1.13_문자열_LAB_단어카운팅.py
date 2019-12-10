#'Yesterday'라는 문자 카운팅하
f = open("C:/Users/student/KMH/Image-analysis-and-develope/python/20191210/yesterday.txt",'r')
yesterday_lyric = f.readlines()
song = yesterday_lyric
f.close()

print(len(song))
print(song)
count = 0
for i in song:
    trans = i.lower()
    count += trans.count('yesterday')
    

print("Number of a Word 'Yesterday' : ", count)
