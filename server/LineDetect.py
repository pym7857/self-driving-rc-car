import cv2
import numpy as np

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
    # 이미지와 color로 채워진 ROI를 합침
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image

# 허프변환 라인 그리기 (확인용)
def draw_lines(img, lines, color=[255, 255, 0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
 
# 허프 변환
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    if (lines is None):
        print("err")
        return "err"
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)

    return lines
 
# 두 이미지 operlap 하기
def weighted_img(img, initial_img, α=1, β=1., λ=0.): 
    return cv2.addWeighted(initial_img, α, img, β, λ)

# '대표선' 구하기 
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

 