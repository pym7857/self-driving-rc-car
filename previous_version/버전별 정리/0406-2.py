from control_thread import *
from math import *
from motor import *
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import tensorflow 

# =========================== 

def grayscale(img): # 흑백이미지로 변환

    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

 

def canny(img, low_threshold, high_threshold): # Canny 알고리즘

    return cv2.Canny(img, low_threshold, high_threshold)

 

def gaussian_blur(img, kernel_size): # 가우시안 필터

    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

 

def region_of_interest(img, vertices, color3=(255,255,255), color1=255): # ROI 셋팅

 

    mask = np.zeros_like(img) # mask = img와 같은 크기의 빈 이미지

    

    if len(img.shape) > 2: # Color 이미지(3채널)라면 :

        color = color3

    else: # 흑백 이미지(1채널)라면 :

        color = color1

        

    # vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 color로 채움 

    cv2.fillPoly(mask, vertices, color)
    #cv2.fillPoly(img, vertices, color)
    cv2.imshow('roi',img)
    # 이미지와 color로 채워진 ROI를 합침

    ROI_image = cv2.bitwise_and(img, mask)

    return ROI_image

 

# 허프변환 라인 그리기 (확인용)

def draw_lines(img, lines, color=[255, 255, 0], thickness=2):

    for line in lines:

        for x1,y1,x2,y2 in line:

            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

            #cv2.imshow('result2', img)

 

# 허프 변환

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap): 
    print()
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    draw_lines(line_img, lines)

 

    return lines

    #return line_img

 

# 두 이미지 operlap 하기

def weighted_img(img, initial_img, α=1, β=1., λ=0.): 

    return cv2.addWeighted(initial_img, α, img, β, λ)

 

# '대표선' 구하기 (62, 63줄 코드 이해 X )

def get_fitline(img, f_lines): 

    lines = np.squeeze(f_lines)

    if(len(lines.shape) == 1): #선이 하나밖에 없는 경우 배열의 모양 따로 조정

        lines = lines.reshape(2,2)

    else:

        lines = lines.reshape(lines.shape[0]*2,2)

    rows,cols = img.shape[:2]

    output = cv2.fitLine(lines,cv2.DIST_L2,0, 0.01, 0.01)

    vx, vy, x, y = output[0], output[1], output[2], output[3]

    x1, y1 = int(((img.shape[0]/2+70)-y)/vy*vx + x) , int(img.shape[0]/2+70)

    x2, y2 = int(((img.shape[0]/2-25)-y)/vy*vx + x) , int(img.shape[0]/2-25)

    

    result = [x1,y1,x2,y2]

    return result


# '대표선' 그리기

def draw_fit_line(img, lines, color=[255, 0, 0], thickness=10):

        # cv2.line(이미지, 시작좌표(0, 0), 끝좌표(500, 500), 색깔, 두께)

        cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), color, thickness) 

 

# '대표선' 이용해서 -> 소실점 구하기 (공식)

def expression(x1,y1,x2,y2,x3,y3,x4,y4):

    m_a = (y2 - y1) / (x2 -x1)

    m_b = (y4 - y3) / (x4 - x3)

    n_a = -((y2 - y1) / (x2 - x1) * x1 ) + y1

    n_b = -((y4 - y3) / (x4 -x3) * x3 ) + y3

    x = (n_b - n_a) / (m_a - m_b) 

    y = m_a * ((n_b - n_a) / (m_a - m_b)) + n_a 

    return x,y

 
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('test.h5')

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
    #elif prediction[0][2] >= 0.9: # LEFT
     #   judge = 'left'
    #elif prediction[0][3] >= 0.9: # HUMAN
     #   judge = 'human'
    #elif prediction[0][4] >= 0.9: # NONE
     #   judge = 'none'
        
    if judge == 'slow':
        print('none!')
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
    
    else: # not none
        if judge == 'stop':
            print('stop!')
            setdrive(S, 0.1)
        elif judge == 'slow':
            print('slow!')
        #elif judge == 'left':
         #   print('left!')
          #  setTurnDrive(FL,3)
           # sleep(3) # wait
        #elif judge == 'human':
         #   print('human!')
          #  setdrive(S, 0.1)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    fps.update()

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
fvs.stop()