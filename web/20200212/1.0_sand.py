listtitle = ['개구리1', '개구리2', '개구리3']
listimg = ['img1.jpg', 'img2.jpg', 'img3.jpg']


listdata = []
# for i,j in zip(listtitle,listimg):
#     print(i,j)
#     list_dict = {'title':i , 'img':j}
#     listdata.append((list_dict))

list(map(lambda x: listdata.append(x), [{'id':n , 'title': i, 'img': j} for n,i,j in zip(range(len(listtitle) , listtitle, listimg))]))




print(listdata)