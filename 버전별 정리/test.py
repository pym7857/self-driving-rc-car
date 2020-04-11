import cv2
import numpy as np
from motor2 import * 
# ===========================================================================================================================
#                                   HOUGHLINE
# ===========================================================================================================================
def mark_img(img, blue_threshold=155, green_threshold=155, red_threshold=155): # 1. 흰색 차선 찾기
    mark = img
    
    #  BGR 제한 값 (기준)
    bgr_threshold = [blue_threshold, green_threshold, red_threshold]

    # BGR 제한 값(기준)보다 작으면 검은색으로
    thresholds = (img[:,:,0] < bgr_threshold[0]) \
                | (img[:,:,1] < bgr_threshold[1]) \
                | (img[:,:,2] < bgr_threshold[2])
    mark[thresholds] = [0,0,0] # 검정색으로 
    return mark
 
def grayscale(img): # 흑백이미지로 변환
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold): # Canny 알고리즘
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices, color3=(255,255,255), color1=255): # ROI 셋팅
    mask = np.zeros_like(img) # mask = img와 같은 크기의 빈 이미지
    
    if len(img.shape) > 2: # Color 이미지(3채널)라면 :
        color = color3
    else: # 흑백 이미지(1채널)라면 :
        color = color1
        
    # vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 color로 채움 
    cv2.fillPoly(mask, vertices, color)
    # 이미지와 color로 채워진 ROI를 합침
    ROI_image = cv2.bitwise_and(img, mask)

    return ROI_image

# 허프변환 라인 그리기 (확인용)
def draw_lines(img, lines, color=[255, 255, 0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)\
            
# 허프 변환
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap): 
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)

    return lines

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

def init():
    global F_FLAG, FR_FLAG, FL_FLAG, f_count, fl_count, fr_count
    F_FLAG = 0
    FL_FLAG = 0
    FR_FLAG = 0
    f_count = 0
    fl_count = 0
    fr_count = 0
    
MOTTIME = 0.001
F_FLAG = 0
FL_FLAG = 0
FR_FLAG = 0
TURN_FLAG = 0
f_count = 0
fl_count = 0
fr_count = 0

height, width = 480, 640
m_width = int(width/2)
'''
vertices = np.array([[(0,height),
                      (0,height-90),
                      (width/2-200, height/2+50),
                      (width/2+200, height/2+50),
                      (width,height-90),
                      (width,height)]], dtype=np.int32)
'''
vertices = np.array([[(0,height),
                      (0,height/2),
                      (width,height/2),
                      (width,height)]], dtype=np.int32)
rho = 1
theta = 1 * np.pi/180
threshold = 30
first_stop = 0

cap = cv2.VideoCapture(0) # 1920x1080 -> 에러
# ===========================================================================================================================
#                                   PLAY
# ===========================================================================================================================
while(cap.isOpened()):
    ret,image = cap.read()
        
    mrk_img = mark_img(image)
    gray_img = grayscale(mrk_img)
    canny_img = canny(gray_img, 70, 210) # Canny edge 알고리즘
    ROI_img = region_of_interest(canny_img, vertices, (0, 0, 255)) # ROI 설정
    line_arr = hough_lines(ROI_img, rho, theta, threshold, 10, 20) # 허프 변환
    
    pre_line_arr = line_arr
    
    if(len(line_arr) == 0):
        line_arr = pre_line_arr

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
    
    pre_line_check = True
    
    if(len(L_lines) == 0 and len(R_lines) == 0): #L_lines, R_lines 모두 없는 경우
        L_lines = pre_left_line
        R_lines = pre_right_line
    elif(len(L_lines) == 0):#L_lines만 없는 경우
        L_lines = pre_left_line
        pre_right_line = R_lines
        print("turn left")
        fl_count += 1
        pre_line_check = False
        if (fl_count == 2):
            init()
            FL_FLAG = 1
            TURN_FLAG = 1
    elif(len(R_lines) == 0):#R_lines만 없는 경우
        pre_left_line = L_lines
        R_lines = pre_right_line
        print("turn right")
        fr_count += 1
        pre_line_check = False
        if (fr_count == 2):
            init()
            FR_FLAG = 1
            TURN_FLAG = 1
    else:#라인 모두 검출한 경우
        pre_right_line = R_lines
        pre_left_line = L_lines
        pre_line_check = True
        #print("LR YES")

    temp = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    
# 왼쪽, 오른쪽 각각 대표선 구하기
    left_fit_line = get_fitline(image,L_lines)
    right_fit_line = get_fitline(image,R_lines)
    #print(left_fit_line) # [158, 539, 388, 370] -> 단 1개 검출 

# 대표선 '그리기'
    draw_fit_line(temp, left_fit_line)
    draw_fit_line(temp, right_fit_line)
    #print('left_fit_line = ', left_fit_line) # [158, 539, 388, 370]

    vanishing_point = expression(left_fit_line[0],left_fit_line[1],left_fit_line[2],left_fit_line[3],right_fit_line[0],right_fit_line[1],right_fit_line[2],right_fit_line[3])
    #print(vanishing_point) # (476.9880952380953, 304.61309523809524)...

    v_x = int(vanishing_point[0])
    v_y = int(vanishing_point[1])

    result = weighted_img(temp, image) # 원본 이미지(=image)에 검출된 선(=temp) overlap
    cv2.circle(result, (v_x,v_y), 6, (0,0,255), -1) # cv2.circle(image, center_coordinates, radius, color, thickness)
    #circle 기준선(보조선)
    cv2.line(result,(m_width,0),(m_width,300),(255,255,0),5) # cv2.line(image, start_point, end_point, color, thickness)

    if(v_x > m_width+140 and (FL_FLAG != 1 and FR_FLAG != 1)): # 소실점의 x좌표가 중앙선보다 오른쪽에 있을때
        print("Right!!!")
        init()
        setdrive(FR, MOTTIME)
    elif(v_x < m_width-140 and (FL_FLAG != 1 and FR_FLAG != 1)): # 소실점의 x좌표가 중앙선보다 왼쪽에 있을때
        print("Left!!!")
        init()
        setdrive(FL, MOTTIME)
    elif(pre_line_check == True):
        print("foward!!!")
        f_count += 1
        if (f_count == 1):
            init()
            F_FLAG = 1
    
    if(first_stop <= 50):
        print('first stop')
        first_stop += 1
    elif(first_stop > 50):
        if(F_FLAG==1):
            if(TURN_FLAG==0):
                setdrive(F, MOTTIME)
            else:
                setdrive(F, MOTTIME)
        elif(FR_FLAG==1):
            setdrive(FR, MOTTIME)
        elif(FL_FLAG==1):
            setdrive(FL, MOTTIME)
    
    #cv2.imshow('ROI', ROI_img)
    cv2.imshow('result2',result) # 결과 이미지 출력
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()