import socket
import cv2
import numpy
from queue import Queue
from _thread import *
from time import *

import RPi.GPIO as GPIO
from time import sleep

# ====================== dist sensor ======================
GPIO.setmode(GPIO.BCM)

GPIO_TRIGGER = 19  
GPIO_ECHO = 26

print("Ultrasonic Distance Measurement")

# 초음파를 내보낼 트리거 핀은 출력 모드로, 반사파를 수신할 에코 피은 입력 모드로 설정한다.
GPIO.setup(GPIO_TRIGGER,GPIO.OUT)
GPIO.setup(GPIO_ECHO,GPIO.IN)
# ====================== dist sensor ======================

# Motor state
STOP = 0
FORWARD = 1
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
FS = 5


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
ENRD = 25
ENLU = 6
# GPIO PIN
IN1 = 16
IN2 = 12# Left Down
IN3 = 4
IN4 = 17 # Right Up
IN5 = 21
IN6 = 20  # Right Down
IN7 = 27
IN8 = 22 # Left Up

# GPIO Library Setting
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

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
def setMotorControl(pwm, INO, INF, speed, stat):
    # Motor speed control to PWM
    pwm.ChangeDutyCycle(speed)
    
    # Forward
    if stat == FORWARD:
        GPIO.output(INO, HIGH)
        GPIO.output(INF, LOW)
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


# Drive algorithm
def setdrive(drv,T):
    if drv == S:
        setMotor(CHLU, 80, STOP) #setSpeed=80
        setMotor(CHLD, 80, STOP)
        setMotor(CHRU, 80, STOP)
        setMotor(CHRD, 80, STOP)
        sleep(T)
    elif drv == F:
        setMotor(CHLU, 60, FORWARD)
        setMotor(CHLD, 60, FORWARD)
        setMotor(CHRU, 60, FORWARD)
        setMotor(CHRD, 60, FORWARD)
        sleep(T)
    elif drv == FL:
        setMotor(CHLU, 50, STOP)
        setMotor(CHLD, 30, FORWARD)
        setMotor(CHRU, 65, FORWARD)
        setMotor(CHRD, 65, FORWARD)
        sleep(T)
    elif drv == FR:
        setMotor(CHLU, 65, FORWARD)
        setMotor(CHLD, 65, FORWARD)
        setMotor(CHRU, 50, STOP)
        setMotor(CHRD, 30, FORWARD)
        sleep(T)
    elif drv == FS:
        setMotor(CHLU, 45, FORWARD)
        setMotor(CHLD, 45, FORWARD)
        setMotor(CHRU, 45, FORWARD)
        setMotor(CHRD, 45, FORWARD)
        sleep(T)

cap = cv2.VideoCapture(0) # 960x540

#cap = cv2.VideoCapture('testvideo.mp4') # 1920x1080 -> 에러
def threaded(): 
    setdrive(F,5)
    setdrive(S,3)
    setdrive(F,5)
start = time()
start_new_thread(threaded,())

while(cap.isOpened()):
    if(time() - start > 5 and time() - start < 8):
        print('human')
    else:
        print('None')
    ret,image = cap.read()    
    cv2.imshow('image',image) # 결과 이미지 출력

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
