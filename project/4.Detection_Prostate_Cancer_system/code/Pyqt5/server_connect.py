import socket
import numpy as np
import cv2
import pickle
import struct
import io
import time
import zlib

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf



def aws_connect(file_path):
    # 서버의 주소입니다. hostname 또는 ip address를 사용할 수 있습니다.
    HOST= 'ec2-13-124-193-28.ap-northeast-2.compute.amazonaws.com'
    # 서버에서 지정해 놓은 포트 번호입니다.
    PORT= 8895

    img = cv2.imread(file_path)
    print("사진크기:{}".format(img.shape))
    encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
    img = cv2.imencode('.png', img, encode_param)
    data = pickle.dumps(img, protocol=3)
    # data = pickle.dumps(img, 0)
    size = len(data)
    # print(size)

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection = client_socket.makefile('wb')
    client_socket.connect((HOST, PORT))

    while True:
        client_socket.sendall(struct.pack(">L", size) + data)
        break
        # message = '1'
        # client_socket.send(message.encode())

        # length = recvall(client_socket,16)
        # stringData = recvall(client_socket, int(length))
        # data = np.frombuffer(stringData, dtype='uint8')

        # decimg=cv2.imdecode(data,1)
        # #cv2.imshow('Image',decimg)

        # key = cv2.waitKey(1)
        # if key == 27:
        #     break
    data = client_socket.recv(1024)

    print("ISUP 점수 값 : {}".format(data.decode()))
    print('close socket')
    client_socket.close()

    return data.decode()

def aws_connect2(file_path):
    # 서버의 주소입니다. hostname 또는 ip address를 사용할 수 있습니다.
    HOST= 'ec2-13-124-193-28.ap-northeast-2.compute.amazonaws.com'
    # 서버에서 지정해 놓은 포트 번호입니다.
    PORT= 8896

    img = cv2.imread(file_path)
    print("사진크기:{}".format(img.shape))
    encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
    img = cv2.imencode('.png', img, encode_param)
    data = pickle.dumps(img, protocol=3)
    # data = pickle.dumps(img, 0)
    size = len(data)
    # print(size)

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection = client_socket.makefile('wb')
    client_socket.connect((HOST, PORT))

    while True:
        client_socket.sendall(struct.pack(">L", size) + data)
        break
        # message = '1'
        # client_socket.send(message.encode())

        # length = recvall(client_socket,16)
        # stringData = recvall(client_socket, int(length))
        # data = np.frombuffer(stringData, dtype='uint8')

        # decimg=cv2.imdecode(data,1)
        # #cv2.imshow('Image',decimg)

        # key = cv2.waitKey(1)
        # if key == 27:
        #     break
    
    data = client_socket.recv(1024)

    print("ISUP 점수 값 : {}".format(data.decode()))
    print('close socket')
    client_socket.close()

    return data.decode()