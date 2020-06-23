import time
from time import sleep
import RPi.GPIO as GPIO


GPIO.setmode(GPIO.BCM)

GPIO_TRIGGER = 19  
GPIO_ECHO = 26

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
    pwm = GPIO.PWM(EN, 10)
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
print("Ultrasonic Distance Measurement")

# 초음파를 내보낼 트리거 핀은 출력 모드로, 반사파를 수신할 에코 피은 입력 모드로 설정한다.
GPIO.setup(GPIO_TRIGGER,GPIO.OUT)
GPIO.setup(GPIO_ECHO,GPIO.IN)

try:
    while True:
        stop = 0
        start = 0
        # 먼저 트리거 핀을 OFF 상태로 유지한다
        GPIO.output(GPIO_TRIGGER, False)
        time.sleep(0.1)

        # 10us 펄스를 내보낸다. 
        # 파이썬에서 이 펄스는 실제 100us 근처가 될 것이다.
        # 하지만 HC-SR04 센서는 이 오차를 받아준다.
        GPIO.output(GPIO_TRIGGER, True)
        time.sleep(0.000001)
        GPIO.output(GPIO_TRIGGER, False)

        # 에코 핀이 ON되는 시점을 시작 시간으로 잡는다.
        while GPIO.input(GPIO_ECHO)==0:
            start = time.time()

        # 에코 핀이 다시 OFF되는 시점을 반사파 수신 시간으로 잡는다.
        while GPIO.input(GPIO_ECHO)==1:
            stop = time.time()

        # Calculate pulse length
        elapsed = stop-start

        # 초음파는 반사파이기 때문에 실제 이동 거리는 2배이다. 따라서 2로 나눈다.
        # 음속은 편의상 340m/s로 계산한다. 현재 온도를 반영해서 보정할 수 있다.
        if (stop and start):
            distance = elapsed * 17000
            print("Distance : %.1f cm" % distance)
    
        if(distance < 2):
            setdrive(S,1)
        else:
            setdrive(F,0.00001)
        
except KeyboardInterrupt:
    print("Ultrasonic Distance Measurement End")
    GPIO.cleanup()

GPIO.cleanup()
