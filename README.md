# ***Self-driving RC CAR:red_car:*** [![HitCount](http://hits.dwyl.com/yudazzi/AutoDrive.svg)](http://hits.dwyl.com/yudazzi/AutoDrive)
<br> ***This project is to OOOO***

## ***0. Team member***

> Contributors/People

| Juyeol RYu | Lee | YoungMin Park |
| :---: | :---: | :---: |
| <img src="https://avatars2.githubusercontent.com/u/49298852?s=460&v=4" width="50%"></img> | <img src="https://avatars1.githubusercontent.com/u/64255265?s=460&v=4" width="50%"></img>  | <img src="https://avatars2.githubusercontent.com/u/44596598?s=460&v=4" width="50%"></img>  |
| ***https://github.com/yudazzi*** | ***https://github.com/*** | ***https://github.com/pym7857*** |   

- ***You can see team member and github profile***
- ***You should probably find team member's lastest project***

## ***1. Development Environment***
* ***OS : Windows 10***
* ***IDE : Pycharm***
* ***Language : Python 3.6***

## ***2. Structure drawing***
![bluePrint](./SampleImages/bluePrint.png) 

## ***3. Hough Line Transform***
허프 변환이란, 이미지 속에서 주요 특징 요소(직선, 원, 타원)들을 찾는 방법입니다. 
다시말해, 이미지 상의 특정한 점들간의 연관성을 찾아 특징을 추출하는 방법입니다.

![image](https://user-images.githubusercontent.com/49298852/85352442-f1f32680-b540-11ea-8fa2-2472f3ed1407.png)

 위 그림을 보면, 왼쪽의 xy평면 상에 기울기 a1과 y절편으로 b1을 갖는 직선 y = a1x + b1이 있고,
이 직선 상의 점 중 임의의 점 (y1, x1), (y2, x2), (y3, x3)가 있습니다.
이 점들을 xy평면 상에서 기울기와 y절편의 평면인 ab평면 상으로 옮기게 되면, 각각 하나의 직선을 갖게 되고, 총 세개의 직선이 나오게 됩니다.

 이때, ab평면이 직선의 기울기와 y절편이라는 것을 생각해보면, 같은 직선 상의 점들은 당연히 같은 기울기와 같은 y절편을 갖게 되므로, 교점을 형성하게 됩니다.

 위 그림의 경우, 같은 직선 상에서 임의의 세점을 뽑았으므로, 당연히 ab평면 상에서 세직선이 하나의 교점을 갖게 되고, 그 좌표 값은 바로 (a1, b1)이 됩니다.
이 좌표값을 가지고 ab평면에서 다시 xy평면으로 바꾸게 되면, 기울기와 y절편을 알고 있으므로 하나의 직선을 구할 수 있게 됩니다.

 위와 같은 원리로 xy 평면 상에서 같은 직선인지 아닌지 모르는 임의의 점들을, ab평면상으로 바꾸어 매핑한 것입니다. ab평면 상에서의 직선을 구하고 그 직선들간의 교점의 존재 여부를 확인해 보면
같은 직선 상의 점인지 아닌지를 알 수 있습니다. 이러한 과정이 바로 Hough transform의 원리입니다.
## ***4. Object Detection***
영상에서 객체 인식을 위한 딥러닝 알고리즘은 영상과 음성분석에서 좋은 성능을 보이는 컨벌류션 뉴럴 네트워크(Convolutional Neural Network, CNN)을 사용했습니다. CNN 이전의 학습은 raw 데이터를 직접 처리하기 때문에 많은 양의 학습데이터가 필요했지만 CNN은 이미지를 컨볼루션 필터를 적용해서 이미지의 특징을 뽑은 feature map을 만듭니다. 과정은 아래와 같이 이루어집니다. 그리고 이 필터에는 가로 세로의 특징을 따로 뽑아낼 수 있는 필터가 각각 존재합니다. 하지만 CNN에서는 이러한 특징을 뽑아주는 가장 적합한 필터를 신경망에서 학습을 통해 자동으로 생성해줍니다.

![image](https://user-images.githubusercontent.com/49298852/85352704-852c5c00-b541-11ea-90d4-4a3357cfca9a.png)

또한 이미지는 보통 RGB 세개의 채널로 구성이 되어있는데 보통은 연산량과 오차를 줄이기 위해서 이미지를 흑백(1채널)으로 바꿔서 처리하지만 color 이미지(3채널)로 처리하기도 합니다.

![image](https://user-images.githubusercontent.com/49298852/85352726-983f2c00-b541-11ea-853c-4bd6867229f7.png)

위의 과정과 같이 필터를 적용하게 되면 input data 보다 out data의 크기가 작아지게 되는데 이를 방지하기 위해서 padding 이라는 기법이 사용됩니다. 이는 필터를 거친 이미지를 0으로 감싸서 특징에는 영향을 미치지 않게 하는 방법입니다. 하지만 이와 같이 이미지의 크기를 유지한채 다음 레이어로 간다면 연산량이 너무 많아지게 됩니다. 때문에 적당한 크기로 줄이고 특징을 강조할수 있도록 pooling 레이어를 사용합니다. 보통 CNN에서는 가장 큰 값을 사용하는 Max Pooling을 사용한다.
CNN의 전체적인 구조는 아래와 같습니다. 특징 추출 단계인 feature extraction 에서의 특징을 추출하는 convolution과 이미지의 크기를 줄이는 pooling 레이어가 존재합니다. 그리고 이미지 분류 단계인 classification에서 인식결과를 얻어냅니다.

![image](https://user-images.githubusercontent.com/49298852/85352758-a725de80-b541-11ea-850e-18b7ea00df92.png)

## ***5. Multi Thread Frame Buffer***
처음에 Multi Thread Frame Buffer를 사용하지 않고 영상을 재생했을 때 조금씩의 끊김이 발생했고 HoughLine Transform 과 Object Detection의 기능이 추가 되자 영상의 끊김이 더욱 커지게 되었습니다. 이러한 문제가 발생한 원인이 영상을 받아들이고 데이터로 가공한 뒤 재생하는 순차적인 과정이 빠르게 이루어지지 못해서 발생했다고 생각했습니다. 그래서 영상을 받아들여서 데이터로 가공해서 큐에 넣는 애니메이션 스레드와 큐에 있는 영상을 꺼내서 재생하는 메인 스레드로 나누어서 실행했습니다. 그 결과 영상 재생 fps가 50%가량 향상되었습니다.

![image](https://user-images.githubusercontent.com/49298852/85352811-c4f34380-b541-11ea-807e-a1435f004582.png)

![image](https://user-images.githubusercontent.com/49298852/85352834-d1779c00-b541-11ea-8618-58c7e27bd06b.png)

## ***6. Socket Programming***
소켓(Socket)은 사전적으로 "구멍", "연결", "콘센트" 등의 의미를 가집니다. 주로 전기 부품을 규격에 따라 연결할 수 있게 만들어진, 구멍 형태의 연결부를 일컫는 단어입니다. 네트워크 프로그래밍에서의 소켓(Socket)에 대한 의미도, 사전적 의미를 크게 벗어나지 않습니다. 프로그램이 네트워크에서 데이터를 송수신할 수 있도록, 네트워크 환경에 연결할 수 있게 만들어진 연결부가 바로 네트워크 소켓(Socket)입니다.

 클라이언트 소켓(Client Socket)과 서버 소켓(Server Socket)의 역할을 먼저 알아야합니다. 데이터를 주고받기 위해서는 먼저 소켓의 연결 과정이 선행되어야 하고, 그 과정에서의 연결 요청과 수신이 각각 클라이언트 소켓과 서버 소켓의 역할입니다.

 두 개의 시스템이 소켓을 통해 네트워크 연결(Connection)을 만들기 위해서는, 최초 어느 한 곳에서 그 대상이 되는 곳으로 연결을 요청해야 합니다. IP 주소와 포트 번호로 식별되는 대상에게, 자신이 데이터 송수신을 위한 네트워크 연결을 수립할 의사가 있음을 알려야 합니다.

 하지만, 최초 한 곳에서 무작정 연결을 시도한다고 해서, 그 요청이 무조건 받아들여지고 연결이 만들어져 데이터를 주고 받을 수 있는 것은 아닙니다. 한 곳에서 연결 요청을 보낸다고 하더라도 그 대상 시스템이 그 요청을 받아들일 준비가 되어 있지 않다면, 해당 요청은 무시되고 연결은 만들어지지 않습니다.
그러므로 요청을 받아들이는 곳에서는 어떤 연결 요청(일반적으로 포트 번호로 식별)을 받아들일 것인지를 미리 시스템에 등록하여, 요청이 수신되었을 때 해당 요청을 처리할 수 있도록 준비해야 합니다.

 이렇듯 두 개의 시스템(또는 프로세스)이 소켓을 통해 데이터 통신을 위한 연결(Connection)을 만들기 위해서는, 연결 요청을 보내는지 또는 요청을 받아들이는지에 따라 소켓의 역할이 나뉘게 되는데, 전자에 사용되는 소켓을 클라이언트 소켓(Client Socket), 후자에 사용되는 소켓을 서버 소켓(Server Socket)이라고 합니다.
 
 ![image](https://user-images.githubusercontent.com/49298852/85352861-e2c0a880-b541-11ea-95af-e789c4173c9d.png)

## ***7. 차량 제어***
<h3>■ 차량 제어 코드</h3>
<pre><code>import RPi.GPIO as GPIO
from time import sleep
# Motor state
STOP =0
FORWARD =1
# Motor channel
CHLU =0
CHLD =1
CHRU =2
CHRD =3
# Drive state
S =0
F =1
B =2
FR =3
FL =4
FS =5
# PIN input output setting
OUTPUT =1
INPUT =0
# PIN setting
HIGH =1
LOW =0
# Real PIN define
# PWM PIN(BCM PIN)
ENLD =5
ENRU =24
ENRD =25
ENLU =6
# GPIO PIN
IN1 =16
IN2 = 12 # Left Down
IN3 =4
IN4 =17 # Right Up
IN5 =21
IN6 =20  # Right Down
IN7 =27
IN8 =22 # Left Up
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

#Motor control algorithm
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
#Motor control easily
def setMotor(ch, speed, stat):
    if ch == CHLD:
        setMotorControl(pwmLD, IN1, IN2, speed, stat)
    elif ch == CHRU:
        setMotorControl(pwmRU, IN3, IN4, speed, stat)
    elif ch == CHRD:
        setMotorControl(pwmRD, IN5, IN6, speed, stat)
    elif ch == CHLU:
        setMotorControl(pwmLU, IN7, IN8, speed, stat)
#Motor Pin Setting(global var)
pwmLD = setPinConfig(ENLD, IN1, IN2) #in 100Hz
pwmRU = setPinConfig(ENRU, IN3, IN4) #in 100Hz
pwmRD = setPinConfig(ENRD, IN5, IN6) #in 100Hz
pwmLU = setPinConfig(ENLU, IN7, IN8) #in 100Hz

#Drive algorithm
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
</code></pre>
## ***8. 차량 조립***
<h3>■ 하드웨어 기능</h3>
<h4>- 자동차 구동</h4>
 4개의 DC모터로 자동차가 움직이며 바퀴 이동속도의 변화를 통해 전진, 좌회전, 우회전을 구현합니다. DC모터는 모터드라이버와 라즈베리파이의 GPIO를 통해 조정합니다.

<h4>- 실시간 영상 확인</h4>
 라즈베리파이와 연결된 pi camera를 통해 자동차 전면 영상을 확인합니다. 이 실시간 영상은 라즈베리파이에서 데이터로 가공된 후에 소켓을 통해 전송되어 houghLine Transform 과 object Detection이 처리 됩니다.

<h4>- 거리 센서를 통해 물체 인식</h4>
 Object detection을 사용해서 물체를 인식해서 물체에 따른 정해진 행동을 하려 했으나 소켓으로 영상을 전송할 때 약간의 지연이 존재해서 판단이 느려지는 문제 발생했습니다. 이 문제를 해결하기 위해서 우선 거리센서를 통해 전방 일정 거리 안에 물체가 인지하고 차량을 정지합니다. 그리고 객체를 object detection으로 판단한 뒤에 주행하도록 했습니다.

<h3>■ 하드웨어 구성</h3>
<h4>- 라떼 판다</h4>
 카메라로 받아들인 프레임을 소켓으로 데스크톱에 전송하고 초음파 거리 센서를 통해 받아들인 값을 통해 차량을 제어 해주는 소형cpu입니다. 라떼 판다의 사양은 아래와 같습니다.

![image](https://user-images.githubusercontent.com/49298852/85352903-fff57700-b541-11ea-8fbc-d48ec1ff37a2.png)

<h4>- DC모터</h4>
 DC모터는 차량의 바퀴를 움직이게 하는 장치입니다. 모터드라이버는 모터를 제어할수 있게 해주는 장치입니다. DC모터는 NP01S-220 4개, 모터드라이버는 L298N 2개를 활용 하였습니다. 모터드라이버 1개당 2륜을 제어하며 두 제품 사양은 다음과 같습니다.

![image](https://user-images.githubusercontent.com/49298852/85352928-14d20a80-b542-11ea-952e-eab741d176ae.png)
![image](https://user-images.githubusercontent.com/49298852/85352954-21eef980-b542-11ea-8c54-9e9adc918238.png)
 
모터드라이버의 Enable 값은 모터를 어느 정도의 속도로 움직이게 하는지에 대한 동기값 입니다. 이 Enable 값을 라즈베리파이 GPIO를 통해 조정하고 이를 통해 나온 Output 값을 DC모터에 연결하여 DC모터를 구동합니다.

<h4>- 파이카메라</h4>
 Pi camera는 라즈베리파이에 연결 가능하며 라즈베리파이 내부에서 조정 가능합니다. 또한 사진 촬영, 동영상 촬영, 실시간 영상 촬영도 가능합니다. 아래는 파이카메라의 사양입니다. 파이카메라를 차량 상단 중앙 부분에 고정하여 차량 전반부 영상을 촬영 가능하도록 하였습니다. 이 영상으로 도로와 장애물, 표지판을 확인하며 이에 따른 행동 교정을 하도록 합니다.

![image](https://user-images.githubusercontent.com/49298852/85352979-3206d900-b542-11ea-98f8-9053c27f5dfd.png)

<h4>- 초음파 거리 센서</h4>
 초음파 거리센서는 송신부와 수신부로 나뉘어져 있으며, 송신부에서 일정한 시간의 간격을 둔 짧은 초음파 펄스를 방사하고, 대상물에 부딪혀 돌아온 에코신호를 수신부에서 받아 이에 대한 시간차를 기반으로 거리를 산출합니다. 이를 통해 장애물의 유무, 물체의 거리 또는 속도 등을 측정할 수 있습니다.

![image](https://user-images.githubusercontent.com/49298852/85353002-3cc16e00-b542-11ea-8517-b2537373b987.png)
