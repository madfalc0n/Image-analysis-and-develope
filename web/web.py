import socket

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_socket.bind(('localhost', 82))
server_socket.listen(3)
print("listening")

while  True :
    client_socket, addr = server_socket.accept()
    print("accepting")
    data = client_socket.recv(65535)    
    data = data.decode()
    print(data)        
    
    try :    
        headers = data.split("\r\n")
        filename = headers[0].split(" ")[1]

        if '.html' in filename:
            file = open("."+ filename, 'rt', encoding='utf-8')
            html = file.read()    
            header = 'HTTP/1.0 200 OK\r\n\r\n'        
            client_socket.send(header.encode("utf-8"))
            client_socket.send(html.encode("utf-8"))
        elif '.jpg' in filename or '.ico' in filename:         
            client_socket.send('HTTP/1.1 200 OK\r\n'.encode())
            client_socket.send("Content-Type: image/jpg\r\n".encode())
            client_socket.send("Accept-Ranges: bytes\r\n\r\n".encode())
            file = open("." + filename, "rb")            
            client_socket.send(file.read())  
            file.close()               
        else :
            header = 'HTTP/1.0 404 File Not Found\r\n\r\n'        
            client_socket.send(header.encode("utf-8"))
    except Exception as e :
        print(e)         
    client_socket.close()