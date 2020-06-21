import socket
import cv2
import numpy
from queue import Queue
from _thread import *

enclosure_queue = Queue()

# 쓰레드 함수 
def threaded(client_socket, addr, queue): 
    print('\nConnected by :', addr[0], ':', addr[1])
    
    while True: 
        try:
            # 1. receive judge (client socket.recv())
            data ,addr= client_socket.recvfrom(1024)
            print('data = ', data)

            if not data: 
                print('Disconnected by ' + addr[0],':',addr[1])
                break

            # 2. send image from queue (client socket.send())
            stringData = queue.get()
            client_socket.send(str(len(stringData)).ljust(16).encode())
            client_socket.send(stringData)

        except ConnectionResetError as e:

            print('Disconnected by ' + addr[0],':',addr[1])
            break
            
    client_socket.close() 

# 1.get webcam image and 2.put to queue
def webcam(queue, capture):
    while True:
        ret, frame = capture.read()

        if ret == False:
            print('ret error')
            continue
        
        # ========================== send webcam display to desktop ========================== 
        encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
        result, imgencode = cv2.imencode('.jpg', frame, encode_param)

        data = numpy.array(imgencode)
        stringData = data.tostring()

        queue.put(stringData)


# main
HOST = '0.0.0.0'
PORT = 8888

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT)) 
server_socket.listen() 

print('server start..')

capture = cv2.VideoCapture(0)

while True: 

    print('wait')
    client_socket, addr = server_socket.accept()
    
    #connection = client_socket.makefile('wb')
    
    # put to queue
    start_new_thread(webcam, (enclosure_queue, capture,))
    
    # recv & send
    start_new_thread(threaded, (client_socket, addr, enclosure_queue,))

server_socket.close()
client_socket.close()
