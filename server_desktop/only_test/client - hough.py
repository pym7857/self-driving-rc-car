import socket
import copy
import time
from LineDetect import *
from math import *
import numpy as np
import cv2

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

        # =========================== hough =================================
        height = frame.shape[0]
        width = frame.shape[1]
        m_width = width // 2

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
        threshold = 30  # threshold 값이  작으면 그 만큼 기준이 낮아져 많은 직선이 검출될 것이고, 값을 높게 정하면 그만큼 적지만 확실한 직선들만 검출이 될 것이다
        line_arr = hough_lines(ROI_img, rho, theta, threshold, 10, 20)  # 허프 변환

        if (line_arr == "err"):
            print('err')

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

        # 라인한개만 나올경우 예외처리
        if (len(L_lines) == 0):
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



        judge = 'turn-forward'

    #frame = cv2.resize(frame, dsize=(200, 150), interpolation=cv2.INTER_AREA)
    cv2.imshow('frame', frame)

    if my_init == 0:
        s = 0
        while(s<4):
            s += 1
        my_init = 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

client_socket.close()