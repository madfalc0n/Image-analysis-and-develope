#!/usr/bin/env python
# coding: utf-8

# In[1]:


from segmentation import seg

TCP_IP = "ec2-13-124-193-28.ap-northeast-2.compute.amazonaws.com"
TCP_PORT = 8896


# In[ ]:


import socket
import cv2
import matplotlib.pyplot as plt
import struct ## new
import zlib
import sys
import pickle
import numpy as np


s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('Socket created')

s.bind((TCP_IP,TCP_PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')


while True :
    conn,addr=s.accept()
    data = b""
    payload_size = struct.calcsize(">L")
    print("payload_size: {}".format(payload_size))
    while True:
        while len(data) < payload_size:
            print("Recv: {}".format(len(data)))
            data += conn.recv(4096)

        print("Done Recv: {}".format(len(data)))
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        print("msg_size: {}".format(msg_size))
        while len(data) < msg_size:
            data += conn.recv(4096)
#             print(len(data))
        frame_data = data[:msg_size]
        data = data[msg_size:]


        #pick_data=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        pick_data=pickle.loads(frame_data)
        #print(frame)
        #print(np.load(image))
        print(pick_data)
        print(len(pick_data))
        image = cv2.imdecode(pick_data[1], cv2.IMREAD_COLOR)
        #image = cv2.imdecode(pick_data, -1)
        cv2.imwrite("output1.png",image)

        break

    img = seg('/home/lab05/kaggle_dir/project_socket_server/output1.png')
#     print(answer)
    print(img.shape)
    print(type(img))
    
#     img = cv2.imread(answer)
    print("사진크기:{}".format(img.shape))
    encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
    img = cv2.imencode('.png', img, encode_param)
    data = pickle.dumps(img, protocol=3)
    # data = pickle.dumps(img, 0)
    size = len(data)
    conn.sendall(struct.pack(">L", size) + data)
    
#     cv2.imwrite("output2.png",answer)

#     answer = str(answer)

    # 메시지를 전송합니다. 
#     conn.sendall(answer.encode())

    conn.close()
    print("check1")
s.close()
    


# In[ ]:




