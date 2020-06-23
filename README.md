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


또한 이미지는 보통 RGB 세개의 채널로 구성이 되어있는데 보통은 연산량과 오차를 줄이기 위해서 이미지를 흑백(1채널)으로 바꿔서 처리하지만 color 이미지(3채널)로 처리하기도 합니다.

위의 과정과 같이 필터를 적용하게 되면 input data 보다 out data의 크기가 작아지게 되는데 이를 방지하기 위해서 padding 이라는 기법이 사용됩니다. 이는 필터를 거친 이미지를 0으로 감싸서 특징에는 영향을 미치지 않게 하는 방법입니다. 하지만 이와 같이 이미지의 크기를 유지한채 다음 레이어로 간다면 연산량이 너무 많아지게 됩니다. 때문에 적당한 크기로 줄이고 특징을 강조할수 있도록 pooling 레이어를 사용합니다. 보통 CNN에서는 가장 큰 값을 사용하는 Max Pooling을 사용한다.
CNN의 전체적인 구조는 아래와 같습니다. 특징 추출 단계인 feature extraction 에서의 특징을 추출하는 convolution과 이미지의 크기를 줄이는 pooling 레이어가 존재합니다. 그리고 이미지 분류 단계인 classification에서 인식결과를 얻어냅니다.
## ***5. Multi Thread Frame Buffer***
## ***6. Socket Programming***
## ***7. 차량 제어***
## ***8. 차량 조립***
