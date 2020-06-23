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
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('light.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

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
    
    # -------------------------------- tm code --------------------------------
    #turn the image into a numpy array
    image_array = np.asarray(frame)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array
    # run the inference
    prediction = model.predict(data)
    judge = ''
    
    if prediction[0][0] >= 0.9: # STOP
        judge = 'stop'
    elif prediction[0][1] >= 0.9: # SLOW
        judge = 'slow'
    elif prediction[0][2] >= 0.9: # LEFT
        judge = 'left'
    elif prediction[0][3] >= 0.9: # HUMAN
        judge = 'human'
    elif prediction[0][4] >= 0.9: # NONE
        judge = 'none'
        
    
    
    
    if judge == 'none':
        print('none!')
    else: # not none
        if judge == 'stop':
            print('stop!')
            setdrive(S, 0.1)
        elif judge == 'slow':
            print('slow!')
        elif judge == 'left':
            print('left!')
            setTurnDrive(FL,3)
            sleep(3) # wait
        elif judge == 'human':
            print('human!')
            setdrive(S, 0.1)
        
    # display the size of the queue on the frame
    cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) 
    # show the frame and update the FPS counter
    cv2.imshow("Frame", frame)
    
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
