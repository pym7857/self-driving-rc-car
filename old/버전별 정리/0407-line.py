import tensorflow.keras
from PIL import Image, ImageOps
from control_thread import *
from math import *
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# Hough Line Transform
from LineDetect import *

# GPIO Library
import RPi.GPIO as GPIO
from time import sleep

# ----------------------------- motor ---------------------------
# Motor state
STOP = 0
FORWARD = 1
BACKWARD = 2
# Motor channel
CHLU = 0
CHLD = 1
CHRU = 2
CHRD = 3
# Drive state
S = 0
F = 1
B = 2
FR = 3
FL = 4
FO = 5
TL = 6
TR = 7
TO = 8

# PIN input output setting
OUTPUT = 1
INPUT = 0
# PIN setting
HIGH = 1
LOW = 0

# Real PIN define
# PWM PIN(BCM PIN)
ENLD = 5
ENRU = 24
ENRD = 6
ENLU = 25
# GPIO PIN
IN1 = 17
IN2 = 4 # Left Down
IN3 = 16
IN4 = 12 # Right Up
IN5 = 2
IN6 = 3  # Right Down
IN7 = 21
IN8 = 20 # Left Up

# GPIO Library Setting
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Servo PIN setting
GPIO.setup(18, GPIO.OUT)
GPIO.setup(13, GPIO.OUT)

# PIN setting algorithm
def setPinConfig(EN, INF, INO): # EN, OFF, ON
    GPIO.setup(EN, GPIO.OUT)
    GPIO.setup(INF, GPIO.OUT)
    GPIO.setup(INO, GPIO.OUT)
    # Activate PWM in 100khz
    pwm = GPIO.PWM(EN, 100)
    # First, PWM is stop
    pwm.start(0)
    return pwm
    
# Motor control algorithm
def setMotorControl(pwm, INF, INO, speed, stat):
    # Motor speed control to PWM
    pwm.ChangeDutyCycle(speed)
    
    # Forward
    if stat == FORWARD:
        GPIO.output(INO, HIGH)
        GPIO.output(INF, LOW)
    # BACKWORD
    elif stat == BACKWARD:
        GPIO.output(INO, LOW)
        GPIO.output(INF, HIGH)
    # STOP
    elif stat == STOP:
        GPIO.output(INO, LOW)
        GPIO.output(INF, LOW)

# Motor control easily
def setMotor(ch, speed, stat):
    if ch == CHLD:
        setMotorControl(pwmLD, IN1, IN2, speed, stat)
    elif ch == CHRU:
        setMotorControl(pwmRU, IN3, IN4, speed, stat)
    elif ch == CHRD:
        setMotorControl(pwmRD, IN5, IN6, speed, stat)
    elif ch == CHLU:
        setMotorControl(pwmLU, IN7, IN8, speed, stat)

# Motor Pin Setting(global var)
pwmLD = setPinConfig(ENLD, IN1, IN2) #in 100Hz
pwmRU = setPinConfig(ENRU, IN3, IN4) #in 100Hz
pwmRD = setPinConfig(ENRD, IN5, IN6) #in 100Hz
pwmLU = setPinConfig(ENLU, IN7, IN8) #in 100Hz
#print('ENLU, ENLD, ENRU, ENRD = ',ENLU, ENLD, ENRU, ENRD)

# Servo algorithm
LU = GPIO.PWM(13,50)
RU = GPIO.PWM(18, 50)
LU.start(0)
RU.start(0)

def setturn(tn,T):
    if tn == TL:
        LU.ChangeDutyCycle(5)
        RU.ChangeDutyCycle(5)
        sleep(T)
    elif tn == TR:
       LU.ChangeDutyCycle(8)
       RU.ChangeDutyCycle(7.7)
       sleep(T)
    elif tn == TO:
       LU.ChangeDutyCycle(6.5)
       RU.ChangeDutyCycle(5.8)
       sleep(T)
#LU  13
#(L)5 ~ 6.37 ~ 7.7(R)
#RU  18
#(L)4.5 ~ 5.8 ~ 7.5(R)

# Drive algorithm
def setdrive(drv,T):
    if drv == S:
        setMotor(CHLU, 80, STOP) #setSpeed=80
        setMotor(CHLD, 80, STOP)
        setMotor(CHRU, 80, STOP)
        setMotor(CHRD, 80, STOP)
        sleep(T)
    elif drv == F:
        setMotor(CHLU, 100, FORWARD)
        setMotor(CHLD, 100, FORWARD)
        setMotor(CHRU, 100, FORWARD)
        setMotor(CHRD, 100, FORWARD)
        sleep(T)

# Drive turn algorithm
def seturndrive(drv,T):
    if drv == FL:
        setdrive(F,0.1)
        setturn(TL,1)
        setdrive(F,T)
    elif drv == FR:
        setdrive(F,0.1)
        setturn(TR,1)
        setdrive(F,T)
    elif drv == FO:
        setdrive(F,0.1)
        setturn(TO,1)
        setdrive(F,T)

#======================================= tm =======================================

#캠 키기
print("[INFO] starting video file thread...")
fvs = FileVideoStream(0).start()
time.sleep(1.0)
# start the FPS timer
fps = FPS().start()

PERMIT_ANGLE = 10 #forward 판단 범위(각도 + -)

while fvs.more():
    frame = fvs.read() 
    #frame = imutils.resize(frame, width=450)
    frame = cv2.resize(frame, dsize=(224,224), interpolation=cv2.INTER_AREA)
    height, width = frame.shape[:2] # 이미지 높이, 너비
    m_width = int(width/2)
    
    
    gray_img = grayscale(frame) # 흑백이미지로 변환
    blur_img = gaussian_blur(gray_img, 3) # Blur 효과

    min_threshold = 70
    max_trheshold = 210

    canny_img = canny(blur_img, min_threshold, max_trheshold) # Canny edge 알고리즘
    #vertices = np.array([[(50,height),(width/2-45, height/2+60), (width/2+45, height/2+60), (width-50,height)]], dtype=np.int32)

    #vertices = np.array([[(0,height/2+30),(width/2-140, height/2-60), (width/2+140, height/2-60), (width,height/2+30)]], dtype=np.int32)
    '''
    vertices = np.array([[(0,height),
                      (0,height/2-25),
                      (width/2-70, height/2-60),
                      (width/2+70, height/2-60),
                      (width,height/2-25),
                      (width,height)]], dtype=np.int32)
    '''
    vertices = np.array([[(0,height), (0, height/2 + 40),
                      (width, height/2 + 40), (width,height)]], dtype=np.int32)
    ROI_img = region_of_interest(canny_img, vertices,(0,0,255)) # ROI 설정

    rho = 1
    theta = 1 * np.pi/180
    threshold = 30    # threshold 값이  작으면 그만큼 기준이 낮아져 많은 직선이 검출될 것이고, 값을 높게 정하면 그만큼 적지만 확실한 직선들만 검출이 될 것이다
    
    line_arr = hough_lines(ROI_img, rho, theta, threshold, 10, 20) # 허프 변환
    line_arr = np.squeeze(line_arr) # remove single dimension (차원을 하나 줄임)

    # 기울기 구하기 (arctan(y,x)이용)
    slope_degree = np.arctan2(line_arr[:,1] - line_arr[:,3], line_arr[:,0] - line_arr[:,2]) * 180 / np.pi
    
# 수평 기울기 제한
    line_arr = line_arr[np.abs(slope_degree)<175]
    slope_degree = slope_degree[np.abs(slope_degree)<175]

# 수직 기울기 제한
    line_arr = line_arr[np.abs(slope_degree)>95]
    slope_degree = slope_degree[np.abs(slope_degree)>95]

# 필터링된 직선 버리기
    L_lines, R_lines = line_arr[(slope_degree>0),:], line_arr[(slope_degree<0),:]
    L_lines, R_lines = L_lines[:,None], R_lines[:,None]

    if(len(L_lines) == 0 and len(R_lines) == 0): #L_lines, R_lines 모두 없는 경우
        L_lines = pre_left_line
        R_lines = pre_right_line

    elif(len(L_lines) == 0):#L_lines만 없는 경우
        L_lines = pre_left_line
        pre_right_line = R_lines

    elif(len(R_lines) == 0):#R_lines만 없는 경우
        R_lines = pre_right_line
        pre_left_line = L_lines

    else:#라인 모두 검출한 경우
        pre_right_line = R_lines
        pre_left_line = L_lines

    temp = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

# 왼쪽, 오른쪽 각각 대표선 구하기
    left_fit_line = get_fitline(frame,L_lines)
    right_fit_line = get_fitline(frame,R_lines)

# 대표선 '그리기'
    draw_fit_line(temp, left_fit_line)
    draw_fit_line(temp, right_fit_line)

    vanishing_point = expression(left_fit_line[0],left_fit_line[1],left_fit_line[2],left_fit_line[3],right_fit_line[0],right_fit_line[1],right_fit_line[2],right_fit_line[3])

    v_x = int(vanishing_point[0])
    v_y = int(vanishing_point[1])

    result = weighted_img(temp, frame) # 원본 이미지(=image)에 검출된 선(=temp) overlap
    cv2.circle(result, (v_x,v_y), 6, (0,0,255), -1) # cv2.circle(image, center_coordinates, radius, color, thickness)

    #circle 기준선(보조선)
    cv2.line(result,(m_width,0),(m_width,300),(255,255,0),5) # cv2.line(image, start_point, end_point, color, thickness)
    
    temp_x, temp_y = m_width/2 , height/2
    #각도 구하기
    angle = int(atan2(height - temp_y,m_width - temp_x)*180/pi)
    #print('angle:', angle)
    
    if(angle > 90 + PERMIT_ANGLE):#오른쪽
        angle = angle - 90
    elif(angle < 90 - PERMIT_ANGLE): #왼쪽
        angle = 90 - angle
    else: #foward
        print("angle-forward")

        
    # display the size of the queue on the frame
    cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) 
    # show the frame and update the FPS counter
    cv2.imshow("Frame", result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    fps.update()

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
fvs.stop()
    
#========================================================================
