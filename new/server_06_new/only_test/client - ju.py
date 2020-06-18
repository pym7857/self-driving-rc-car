import socket
import tensorflow.keras
import copy
import time
from LineDetect import *
from math import *
import numpy as np
import cv2

#======================================= tm =======================================
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('aug_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

#======================================= socket setting =======================================
HOST = '192.168.0.115'
PORT = 8888

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

#======================================= tm judge =======================================\
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

start = int(time.time())
prev_second = 5

judge = ' '

my_init = 0

prevTime = 0
while True:
    client_socket.sendto(judge.encode(), (HOST, PORT))

    length = recvall(client_socket, 16)
    stringData = recvall(client_socket, int(length))
    img_data = np.frombuffer(stringData, dtype='uint8')

    frame = cv2.imdecode(img_data, 1)

    if( (int(time.time())-start) % 1 == 0 and prev_second != int(time.time())-start):

        prev_second = int(time.time())-start
        #print(prev_second)

        frame_resize = copy.deepcopy(frame)
        #frame_resize = frame_resize.resize(224, 224, 3)
        frame_resize = cv2.resize(frame, dsize=(224, 224), interpolation=cv2.INTER_AREA)
        # ================================tm-code=============================================
        # turn the image into a numpy array
        image_array = np.asarray(frame_resize)
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array
        # run the inference
        prediction = model.predict(data)

        if prediction[0][0] >= PERCENTAGE:  # STOP
            judge = 'stop'
            if prev == 'stop':
                stop_c += 1
            else:
                prev = 'stop'
                init_count()  # init all count
                stop_c += 1
        elif prediction[0][1] >= PERCENTAGE:  # SLOW
            judge = 'slow'
            if prev == 'slow':
                slow_c += 1
            else:
                prev = 'slow'
                init_count()  # init all count
                slow_c += 1
        elif prediction[0][2] >= PERCENTAGE:  # LEFT
            judge = 'left'
            if prev == 'left':
                left_c += 1
            else:
                prev = 'left'
                init_count()  # init all count
                left_c += 1
        elif prediction[0][3] >= PERCENTAGE:  # HUMAN
            judge = 'human'
            if prev == 'human':
                human_c += 1
            else:
                prev = 'human'
                init_count()  # init all count
                human_c += 1
        elif prediction[0][4] >= PERCENTAGE:  # NONE
            judge = 'none'

        # =========================== hough =================================

        height = frame.shape[0]
        width = frame.shape[1]
        m_width = width//2

        gray_img = grayscale(frame)  # 흑백이미지로 변환
        blur_img = gaussian_blur(gray_img, 3)  # Blur 효과
        min_threshold = 70
        max_trheshold = 210
        canny_img = canny(blur_img, min_threshold, max_trheshold)  # Canny edge 알고리즘

        vertices = np.array([[(0, height), (0, height / 2),
                              (width, height / 2), (width, height)]], dtype=np.int32)
        ROI_img = region_of_interest(canny_img, vertices, (0, 0, 255))  # ROI 설정

        rho = 1
        theta = 1 * np.pi / 180
        threshold = 30  # threshold 값이  작으면 그만큼 기준이 낮아져 많은 직선이 검출될 것이고, 값을 높게 정하면 그만큼 적지만 확실한 직선들만 검출이 될 것이다
        line_arr = hough_lines(ROI_img, rho, theta, threshold, 10, 20)  # 허프 변환

        if (line_arr == "err"):
            print('err')
            #return "err - forward"

        line_arr = np.squeeze(line_arr)  # remove single dimension (차원을 하나 줄임)
        slope_degree = np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180 / np.pi
        line_arr = line_arr[np.abs(slope_degree) < 175]
        slope_degree = slope_degree[np.abs(slope_degree) < 175]

        # 수직 기울기 제한
        line_arr = line_arr[np.abs(slope_degree) > 95]
        slope_degree = slope_degree[np.abs(slope_degree) > 95]

        # 필터링된 직선 버리기
        L_lines, R_lines = line_arr[(slope_degree > 0), :], line_arr[(slope_degree < 0), :]
        L_lines, R_lines = L_lines[:, None], R_lines[:, None]

        #라인한개만 나올경우 예외처리
        if(len(L_lines) == 0):
            judge = 'turn-left'
            continue
        elif (len(R_lines) == 0):
            judge = 'turn-right'
            continue

        temp = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

        # 왼쪽, 오른쪽 각각 대표선 구하기
        left_fit_line = get_fitline(frame, L_lines)
        right_fit_line = get_fitline(frame, R_lines)

        # 대표선 '그리기'
        draw_fit_line(temp, left_fit_line)
        draw_fit_line(temp, right_fit_line)

        vanishing_point = expression(left_fit_line[0], left_fit_line[1], left_fit_line[2], left_fit_line[3],
                                     right_fit_line[0], right_fit_line[1], right_fit_line[2], right_fit_line[3])

        v_x = int(vanishing_point[0])
        v_y = int(vanishing_point[1])

        result = weighted_img(temp, frame)  # 원본 이미지(=image)에 검출된 선(=temp) overlap
        cv2.circle(result, (v_x, v_y), 6, (0, 0, 255),
                   -1)  # cv2.circle(image, center_coordinates, radius, color, thickness)

        # circle 기준선(보조선)
        cv2.line(result, (m_width, 0), (m_width, 300), (255, 255, 0),
                 5)  # cv2.line(image, start_point, end_point, color, thickness)
        temp_x, temp_y = m_width / 2, height / 2
        # 소실점 v_x,v_y 기준점 : m_width,height
        '''
        if (v_x > m_width + 60):  # 소실점의 x좌표가 중앙선보다 오른쪽에 있을때
            print('angle-right-vanishing')
            judge = 'turn-right-vanishing'
        elif (v_x < m_width - 60):  # 소실점의 x좌표가 중앙선보다 왼쪽에 있을때
            print('angle-left-vanishing')
            judge = 'turn-left-vanishing'
        else:
            print("angle-forward")
            judge = 'turn-forward'
        '''
        #print("angle-forward")
        judge = 'turn-forward'
    #frame = cv2.resize(frame, dsize=(200, 150), interpolation=cv2.INTER_AREA)
    curTime = time.time()
    sec = curTime - prevTime
    prevTime = curTime
    fps = 1 / (sec)
    fps_str = "FPS : %0.1f" % fps
    cv2.putText(frame, fps_str, (5,20), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0))
    cv2.imshow('frame', frame)

    if my_init == 0:
        s = 0
        while(s<4):
            s += 1
        my_init = 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

client_socket.close()