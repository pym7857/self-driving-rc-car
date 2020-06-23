import socket
import tensorflow.keras
import time
import numpy as np
import cv2

#======================================= tm =======================================
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('RHN.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

#======================================= socket setting =======================================
HOST = '192.168.0.115'
PORT = 8888
judge = ' '

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))
print('connect success!')

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

start = int(time.time())
prev_second = 5
prev = 'none'
PERCENTAGE = 0.8

while True:
    client_socket.sendto(judge.encode(), (HOST, PORT))

    length = recvall(client_socket, 16)
    stringData = recvall(client_socket, int(length))
    img_data = np.frombuffer(stringData, dtype='uint8')

    frame = cv2.imdecode(img_data, 1)
    frame = cv2.resize(frame, dsize=(224, 224), interpolation=cv2.INTER_AREA)

    if ((int(time.time()) - start) % 1 == 0 and prev_second != int(time.time()) - start):
        prev_second = int(time.time()) - start
        # ================================tm-code=============================================
        # turn the image into a numpy array
        image_array = np.asarray(frame)
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # Load the image into the array
        data[0] = normalized_image_array
        # run the inference
        prediction = model.predict(data)

        if prediction[0][0] >= PERCENTAGE:  # RIGHT
            judge = 'right'
            if prev == 'right':
                left_c += 1
            else:
                prev = 'right'
                init_count()  # init all count
                left_c += 1
        elif prediction[0][1] >= PERCENTAGE:  # HUMAN
            judge = 'human'
            if prev == 'human':
                human_c += 1
            else:
                prev = 'human'
                init_count()  # init all count
                human_c += 1
        elif prediction[0][2] >= PERCENTAGE:  # NONE
            judge = 'none'

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

client_socket.close()