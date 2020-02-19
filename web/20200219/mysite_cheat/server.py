import socket

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_socket.bind(('localhost', 82))
server_socket.listen(0)

client_socket, addr = server_socket.accept()


for i in range(30) :
    data = client_socket.recv(65535)
    print(data.decode())
    