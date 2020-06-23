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
## ***5. Multi Thread Frame Buffer***
## ***6. Socket Programming***
## ***7. 차량 제어***
## ***8. 차량 조립***
