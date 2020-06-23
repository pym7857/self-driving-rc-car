import socket
import tensorflow.keras
import numpy as np
import cv2
import threading
from multi_thread_buffer import *
from time import *
PERMIT_ANGLE = 10 #forward 판단 범위(각도 + -)

stop_c = 0
slow_c = 0
left_c = 0
human_c = 0

def init_count():
    global stop_c, slow_c, left_c, human_c
    stop_c = 0
    slow_c = 0
    left_c = 0
    human_c = 0

prev = 'none'
PERCENTAGE = 0.8
message = 'none'

def send(sock):
    while True:
        if (message == 'stop'):
            print('stop x 5')
            sock.send(message.encode())


def receive(sock):
    global stop_c, slow_c, left_c, human_c, message

    def recvall(sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    while True:
        length = recvall(client_socket, 16)
        stringData = recvall(sock, int(length))  # 2. stringData 받기
        img_data = np.frombuffer(stringData, dtype='uint8')

        frame = cv2.imdecode(img_data, 1)
        cv2.imshow('frame - test', frame)

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = np.dstack([frame, frame, frame])
        # cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ================================tm-code=============================================
        # turn the image into a numpy array
        image_array = np.asarray(frame)
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # Load the image into the array
        data[0] = normalized_image_array
        # run the inference
        prediction = model.predict(data)

        if prediction[0][0] >= PERCENTAGE:  # STOP
            # message = 'stop'
            if prev == 'stop':
                stop_c += 1
            else:
                prev = 'stop'
                init_count()  # init all count
                stop_c += 1
        elif prediction[0][1] >= PERCENTAGE:  # SLOW
            # message = 'slow'
            if prev == 'slow':
                slow_c += 1
            else:
                prev = 'slow'
                init_count()  # init all count
                slow_c += 1
        elif prediction[0][2] >= PERCENTAGE:  # LEFT
            # message = 'left'
            if prev == 'left':
                left_c += 1
            else:
                prev = 'left'
                init_count()  # init all count
                left_c += 1
        elif prediction[0][3] >= PERCENTAGE:  # HUMAN
            # message = 'human'
            if prev == 'human':
                human_c += 1
            else:
                prev = 'human'
                init_count()  # init all count
                human_c += 1
        elif prediction[0][4] >= PERCENTAGE:  # NONE
            message = 'none'

        if stop_c >= 5:
            print('find stop_c == 5 ! ')
            message = 'stop x 5'
            init_count()

        cv2.imshow('frame', frame)

#======================================= tm =======================================
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('aug_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

#======================================socket========================================
HOST = '192.168.0.115'
PORT = 8888

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))

sender = threading.Thread(target=send, args=(client_socket,))
receiver = threading.Thread(target=receive, args=(client_socket,))

receiver.start()
sender.start()


while True:
    sleep(1)
    pass

client_socket.close()
fvs.stop()
#===================================== define ========================================

#=================================== multi thread buffer ===============================
#print('[INFO] starting video file thread..')
#fvs = FileVideoStream(0).start()

