import numpy as np
import cv2
import tensorflow.keras
from PIL import Image, ImageOps
from math import *
from imutils.video import FPS
import numpy as np
import argparse
import time
import socket
import socketserver
import threading

# Hough Line Transform
from LineDetect import *


judge = ''

# ======================================= tm =======================================
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('f_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

PERMIT_ANGLE = 10

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

def houghLine(image):
	global stop_c, slow_c, left_c, human_c, prev, judge

	frame = image
	# frame = imutils.resize(frame, width=450)
	frame = cv2.resize(frame, dsize=(224, 224), interpolation=cv2.INTER_AREA)

	dst = frame.copy()
	dst = dst[92:224, 60:160]
	dst = cv2.resize(dst, dsize=(224, 224), interpolation=cv2.INTER_AREA)

	height, width = frame.shape[:2]  # 이미지 높이, 너비
	m_width = int(width / 2)

	# -------------------------------- tm code --------------------------------
	# turn the image into a numpy array
	image_array = np.asarray(dst)
	# Normalize the image
	normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
	# Load the image into the array
	data[0] = normalized_image_array
	# run the inference
	prediction = model.predict(data)
	judge = ''

	PERCENTAGE = 0.7

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

	if stop_c >= 5:
		print('find stop_c == 5 ! ')

	# =========================== hough =================================
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

	if (len(L_lines) == 0 and len(R_lines) == 0):  # L_lines, R_lines 모두 없는 경우
		L_lines = pre_left_line
		R_lines = pre_right_line
	elif (len(L_lines) == 0):  # L_lines만 없는 경우
		L_lines = pre_left_line
		pre_right_line = R_lines
	elif (len(R_lines) == 0):  # R_lines만 없는 경우
		R_lines = pre_right_line
		pre_left_line = L_lines
	else:  # 라인 모두 검출한 경우
		pre_right_line = R_lines
		pre_left_line = L_lines

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
	# 각도 구하기

	if judge == 'none':
		print('none!')
		# setdrive(F, 0.1)
		angle = int(atan2(height - temp_y, m_width - temp_x) * 180 / pi)
		print('angle= ', angle)

		if (angle > 90 + PERMIT_ANGLE):  # 오른쪽
			angle = angle - 90
			print('angle-right')
		elif (angle < 90 - PERMIT_ANGLE):  # 왼쪽
			angle = 90 - angle
			print('angle-left')
		else:  # foward
			print("angle-forward")
	else:  # not none
		if judge == 'stop':
			print('stop!')
		# setdrive(S, 1)
		elif judge == 'slow':
			print('slow!')
		# setdrive(S, 0.1)
		elif judge == 'left':
			print('left!')
		# setdrive(S, 0.1)
		elif judge == 'human':
			print('human!')
	# setdrive(S, 1)

	# show the frame and update thqe FPS counter
	cv2.imshow("Frame", result)

	return judge



class MyTCPHandler(socketserver.BaseRequestHandler):

	def handle(self):
		global judge
		sock = self.request
		sock.send(judge)

class VideoStreamingTest(object):
	def __init__(self, host, port1, port2):

		self.host = host
		self.port1 = port1
		self.port2 = port2

		print("start")
		self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.server_socket.bind((self.host, self.port1))
		self.server_socket.listen(0)
		self.connection, self.client_address = self.server_socket.accept()
		print("accepted")

		self.data = self.connection.makefile('rwb')
		self.host_name = socket.gethostname()
		self.host_ip = socket.gethostbyname(self.host_name)

		self.streaming()

	def streaming(self):
		global judge
		try:
			print("Host: ", self.host_name + ' ' + self.host_ip)
			print("Connection from: ", self.client_address)
			print("Streaming...")
			print("Press 'q' to exit")

			# need bytes here
			stream_bytes = b' '
			judge = ''
			while True:
				stream_bytes += self.data.read(1024)
				first = stream_bytes.find(b'\xff\xd8') #jpeg start
				last = stream_bytes.find(b'\xff\xd9')  #jpeg end
				if first != -1 and last != -1:
					jpg = stream_bytes[first:last + 2]
					stream_bytes = stream_bytes[last + 2:]
					image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
					judge = houghLine(image)

					print('j =', judge)

					# judge
					server = socketserver.TCPServer((self.host, self.port2), MyTCPHandler)
					print('judge 서버 시작...')
					server.serve_forever()  # 클라이언트로부터 접속 요청을 받아들일 준비

				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

		finally:
			self.connection.close()
			self.server_socket.close()

if __name__ == '__main__':
	h, p1, p2 = "192.168.0.102", 8000, 8002

	# 영상 스트리밍
	VideoStreamingTest(h, p1, p2)

