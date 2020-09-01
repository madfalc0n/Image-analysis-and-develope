#!/usr/bin/env python
# coding: utf-8

# In[1]:


import socket
host = "ec2-13-124-193-28.ap-northeast-2.compute.amazonaws.com"
port = 8895


# In[2]:


server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)

server_socket.bind((host,port))

server_socket.listen()

client_socket, addr = server_socket.accept()

print('connected by',addr)

# 무한루프를 돌면서 
while True:

    # 클라이언트가 보낸 메시지를 수신하기 위해 대기합니다. 
    data = client_socket.recv(1024)

    # 빈 문자열을 수신하면 루프를 중지합니다. 
    if not data:
        break


    # 수신받은 문자열을 출력합니다.
    print('Received from', addr, data.decode())

    # 받은 문자열을 다시 클라이언트로 전송해줍니다.(에코) 
    client_socket.sendall(data)


# 소켓을 닫습니다.
client_socket.close()
server_socket.close()

