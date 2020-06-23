import socket
import numpy as np
import cv2

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

HOST = '192.168.0.115'
PORT = 8888

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))

cnt = 0
while True:

    message = '1'
    client_socket.sendto(str(cnt).encode(), (HOST, PORT))
    '''
    if(cnt%5 == 0):
        client_socket.sendto(str(cnt).encode(),(HOST,PORT))
    else:
        client_socket.sendto(message.encode(), (HOST, PORT))
        
    cnt += 1
    '''
    length = recvall(client_socket, 16)
    stringData = recvall(client_socket, int(length))
    data = np.frombuffer(stringData, dtype='uint8')

    decimg = cv2.imdecode(data, 1)
    cv2.imshow('Image', decimg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

client_socket.close()