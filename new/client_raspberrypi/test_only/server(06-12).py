import socket
import cv2
import numpy
from queue import Queue
from _thread import *

import RPi.GPIO as GPIO
from time import sleep
from time import *

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
        setMotor(CHLU, 100, STOP)
        setMotor(CHLD, 40, FORWARD)
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
        setMotor(CHLU, 70, FORWARD)
        setMotor(CHLD, 70, FORWARD)
        setMotor(CHRU, 70, FORWARD)
        setMotor(CHRD, 70, FORWARD)
        sleep(T)

sleep_hough = False

enclosure_queue = Queue()

start = time()
print('timer sec start~')

# 쓰레드 함수 
def threaded_hough(client_socket, addr, queue): 
    global sleep_hough
    print('Connected by :', addr[0], ':', addr[1])
    
    while True: 
        if(sleep_hough == False):
            try:
                if(int(time()-start) % 1 == 0):
                    # 1. receive judge (client socket.recv())
                    data = client_socket.recv(1024)
                    
                    if not data: 
                        print('Disconnected by ' + addr[0],':',addr[1])
                        break
                    
                    new_data = str(data.decode("utf-8"))
                    
                    #print('new_data=', new_data)
                    if (new_data == 'turn-forward'):
                        setdrive(F, 0.00001)
                    elif (new_data == 'turn-right'):
                        setdrive(FR, 0.00001)
                    elif (new_data == 'turn-left'):
                        setdrive(FL, 0.00001)
              
                    # 2. send image from queue (client socket.send())
                    #if(time()-start > 16):
                        #print('16sec passed')

                    stringData = queue.get()    
                    client_socket.send(str(len(stringData)).ljust(16).encode())
                    client_socket.send(stringData)
                    

            except ConnectionResetError as e:

                print('Disconnected by ' + addr[0],':',addr[1])
                break
        else:
            sleep(10)
            
    client_socket.close() 

def threaded_tm(client_socket, addr, queue): 
    global sleep_hough
    print('Connected by :', addr[0], ':', addr[1])
       
    while True: 

        try:
            if(int(time()-start) % 1 == 0):
                # 1. receive judge (client socket.recv())
                data = client_socket.recv(1024)
                
                if not data: 
                    print('Disconnected by ' + addr[0],':',addr[1])
                    break
                
                new_data = str(data.decode("utf-8"))

                #print('new_data=', new_data)
                if (new_data == 'right'):
                    print('>>right')
                elif (new_data == 'human'):
                    sleep_hough = True
                    print('>>human')
                    setdrive(S,10)
                    sleep(10)
                
                
                # 2. send image from queue (client socket.send())
                #if(time()-start > 16):
                    #print('16sec passed')

                stringData = queue.get()    
                client_socket.send(str(len(stringData)).ljust(16).encode())
                client_socket.send(stringData)
                
                sleep_hough = False
                

        except ConnectionResetError as e:

            print('Disconnected by ' + addr[0],':',addr[1])
            break
            
    client_socket.close()
    
# 1.get webcam image and 2.put to queue
def webcam(queue, capture):

    while True:
        ret, frame = capture.read()

        if ret == False:
            continue


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

server_socket2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
server_socket2.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket2.bind((HOST, 8889)) 
server_socket2.listen() 

print('server start')

capture = cv2.VideoCapture(0)


while True: 

    print('wait')
    client_socket, addr = server_socket.accept()
    client_socket2, addr = server_socket2.accept()
    
    #connection = client_socket.makefile('wb')
    
    # put to queue
    start_new_thread(webcam, (enclosure_queue, capture,))
    
    # recv & send
    start_new_thread(threaded_tm, (client_socket, addr, enclosure_queue,))
    start_new_thread(threaded_hough, (client_socket2, addr, enclosure_queue,))

server_socket.close()
client_socket.close()
client_socket2.close()
GPIO.cleanup()