import math
import cv2
import numpy as np

#Determine the center and radius of the circle to use
def avg_circles(circles, b):
    x=0
    y=0
    r=0
    for i in range(b):
        x = x + circles[0][i][0]
        y = y + circles[0][i][1]
        r = r + circles[0][i][2]
    x = int(x/(b))
    y = int(y/(b))
    r = int(r/(b))
    return x, y, r


# Determine the angle of 2 straight lines created from 4 points
def angle_between_lines(point1_line1, point2_line1, point1_line2, point2_line2):
    # Calculate vectors representing the lines
    vector_line1 = [point2_line1[0] - point1_line1[0], point2_line1[1] - point1_line1[1]]
    vector_line2 = [point2_line2[0] - point1_line2[0], point2_line2[1] - point1_line2[1]]

    dot_product = np.dot(vector_line1, vector_line2)
    magnitude1 = np.linalg.norm(vector_line1)
    magnitude2 = np.linalg.norm(vector_line2)
    cos_theta = dot_product / (magnitude1 * magnitude2)
    
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)

    return angle_deg

# Check if the line has an intersection point with the circle created from the center and the radius passed in
def is_line_intersect_circle(point1, point2, circle_center, radius):
    
    x1, y1 = point1
    x2, y2 = point2
    cx, cy = circle_center

    if x1 == x2:
        return abs(x1 - cx) <= radius and min(y1, y2) <= cy <= max(y1, y2)
    else:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        distance = abs(m * cx - cy + b) / math.sqrt(m**2 + 1)

        return distance <= radius

def fillOutSideCircle(img, center, r):
    # Create a black mask with the same size as the image
    mask = np.zeros_like(img)

    # Draw filled the area outside of circles on the mask
    
    cv2.circle(mask, center, r-6, (255, 255, 255), -1)

    # Apply the mask to the original image
    result = cv2.bitwise_and(img, img, mask=mask)
    
    return result

def get_line_length(line):
    x1, y1, x2, y2 = line
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


# Used to determine circle range and radius
def detection_circle(img):
    image = img
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #Get the center position and radius of the circle
    height, width = image.shape[:2]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, np.array([]), 100, 50, int(height*0.35), int(height*0.48))
    a, b, c = circles.shape
    x,y,r = avg_circles(circles, b)

    return x, y ,r

def getTime(image, x, y ,r):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = 100
    maxValue = 255
    th, gray = cv2.threshold(gray, thresh, maxValue, cv2.THRESH_BINARY_INV)
    
    #Blacken the part that does not belong to the circle
    gray = fillOutSideCircle(gray,(x, y),r)

    cv2.circle(image, (x, y), 2, (0, 255, 0), 3)
    
    blurred_img = cv2.GaussianBlur(image, (5, 5), 0)

    edges = cv2.Canny(blurred_img, 50, 150)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=60, minLineLength=r/3, maxLineGap=5)

    copied_img = np.copy(image)
    
    #Determine the lines to draw
    lineDrawed = []
    
    for i in range(0, len(lines)):
        x1, y1, x2, y2 = lines[i][0]
        if(is_line_intersect_circle((x1, y1), (x2, y2), (x, y), r/4)):
            if(get_line_length((x, y, x2, y2)) > get_line_length((x, y, x1, y1))):
                lineDrawed.append([(x, y),(x2, y2)])
            else:
                lineDrawed.append([(x, y),(x1, y1)])
                
    #Calculate the angle of the lines relative to the line formed by the center of the circle
    angles = []
    for i in lineDrawed:
        (x, y),( x1, y1) = i
        angles.append(angle_between_lines((x, y), (x, y+r), (x, y), (x1, y1)))
    
    #Union of lines and angles. Move the corners to the same clockwise direction.
    angle_with_lines = []
    for i in range(0, len(angles)):
        (x, y),( x1, y1) = lineDrawed[i]
        if (x1 > x):
            angle = 360 - angles[i]
        else:
            angle = angles[i]
        angle_with_lines.append([(x, y),( x1, y1) ,angle, get_line_length((x, y, x1, y1))])
    
    angle_with_lines = sorted(angle_with_lines, key=lambda x: x[2])
    
    k = 0
    distance = []
    linesDraw =[]
    
    #Reduce overlapping lines on clock hands
    while(k < len(angles)):
        if(k >= len(angles)):
            break
        
        temp = []
        temp.append(angle_with_lines[k])
        o = 0
        for i in range(k+1, len(angles)):

            if(angle_with_lines[i][2] - temp[0][2] < 7):
                o = i
                temp.append(angle_with_lines[i])
            else:
                o = i-1
                break
        
        k = o+1
        
        if(len(temp) > 1):
            temp = sorted(temp, key=lambda x: x[3])
            value = np.abs(temp[0][2]- temp[-1][2])
            t= [temp[-1],value]
            distance.append(value)
            linesDraw.append(t)
            
        else:

            distance.append(0)
            t= [temp[0],0]
            linesDraw.append(t)
    
    linesDraw = sorted(linesDraw, key=lambda x: x[0][3], reverse=True)
    
    #Draw frames around the clock hands
    for i in range(0, len(linesDraw)):

        if(i == 2):
            if(linesDraw[2][0][2]/30 > 6):
                hour = int(linesDraw[2][0][2]/30 - 6)
            else:
                hour = int(linesDraw[2][0][2]/30 + 6)
            
            color = (0, 255, 0)
        else:
            if(i == 0 and linesDraw[0][1] > linesDraw[1][1]):
                if(linesDraw[i][0][2]/6 > 30):
                    min = int(linesDraw[i][0][2]/6 - 30)

                else:
                    min = int(linesDraw[i][0][2]/6 + 30)

                
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
                if(i == 1 and linesDraw[1][1] > linesDraw[0][1]):
                    
                    if(linesDraw[i][0][2]/6 > 30):
                        min = int(linesDraw[i][0][2]/6 - 30)

                    else:
                        min = int(linesDraw[i][0][2]/6 + 30)

                    color = (255, 0, 0)
                    
                else:
                    if(linesDraw[i][0][2]/6 > 30):
                        sec = int(linesDraw[i][0][2]/6 - 30)

                    else:
                        sec = int(linesDraw[i][0][2]/6 + 30)

                    color = (0, 0, 255)
                    
        (x, y) = linesDraw[i][0][0]
        ( x1, y1)= linesDraw[i][0][1]
        if(i == 2):
            color = (0, 255, 0)
            
        cv2.rectangle(copied_img, (x, y), (x1, y1), color, 1)
    
    # Write time on image
    text = "Time: " + str(hour) + " : " + str(min) + " : " + str(sec)
    position = (10, 30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    height, width, _ = image.shape
    print(height, width)
    min_width_for_large_image = 1000 
    
    if width > min_width_for_large_image:
        font_scale = 3
        font_color = (0, 0, 0) 
        position = (30, 100)
        thickness = 10
    elif width > 700:
        font_scale = 1.75
        position = (10, 50)
        font_color = (0, 0, 0) 
        thickness = 2
    else:
        font_scale = 0.7
        font_color = (0, 0, 0) 
        thickness = 2

    cv2.putText(copied_img, text, position, font, font_scale, font_color, thickness)
    cv2.imwrite("resultImage.png", copied_img)
    cv2.waitKey()


def main(nameOfIamge):
    try:
        image = cv2.imread(nameOfIamge)
        x, y ,r = detection_circle(image)
        getTime(image,x, y ,r)
    except:
        print("Can't find clock!!!")

#change input image
main('clock1.png')
