from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt

def merge_boxes(boxes):
    # print(boxes)
    # print(len(boxes))
    new_box = {
        'x0': min([b['x0'] for b in boxes]),
        'x1': max([b['x1'] for b in boxes]),
        'top': min([b['top'] for b in boxes]),
        'bottom': max([b['bottom'] for b in boxes]),
        'text': ' '.join([b['text'] for b in boxes])
    }
    # print(new_box)
    
    return new_box

def get_rows(boxes, thresh=2):
    # boxes = sorted(boxes, key=lambda b: b['x0'])
    boxes = sorted(boxes, key=lambda b: b['top'])
    
    checked = [False] * len(boxes)
    rows = []
    
    for i in range(len(boxes)):
        if checked[i]:
            continue
        checked[i] = True
        row = [boxes[i]]
        for j in range(i+1, len(boxes)):
            if checked[j]:
                continue
            if (boxes[j]['bottom'] - boxes[i]['top'] > thresh
                and boxes[i]['bottom'] - boxes[j]['top'] > thresh):
               row.append(boxes[j])
               checked[j] = True
        rows.append(row)
        
    return rows

def strip_row(row, thresh=3.5):
    row = sorted(row, key=lambda b: b['x0'])
    # print(row)
    
    lines = []

    line = [row[0]]
    for i in range(1, len(row)):
        if row[i]['x0'] - row[i-1]['x1'] < thresh:
            line.append(row[i])
        else:
            lines.append(line)
            line = [row[i]]
    lines.append(line)

    return lines

def group_boxes(boxes, thresh=0):
    # print(boxes[:5])
    new_boxes = []
    
    i = 0
    for row in get_rows(boxes):
        # new_boxes.append(merge_boxes(row))
        for line in strip_row(row):
            new_boxes.append(merge_boxes(line))
    # print(len(boxes))
        # break
    
    return new_boxes

def detect_dash(line_img):
    if len(line_img.shape) > 2:
        grey = cv2.cvtColor(line_img,cv2.COLOR_BGR2GRAY)
    else:
        grey = line_img

    grey = cv2.bilateralFilter(grey, 9, 75, 75)
    # grey = cv2.morphologyEx(grey, cv2.MORPH_OPEN, (7,7))
    # grey = cv2.dilate(grey, (9,9), iterations=1)
    edges = cv2.Canny(grey, 120, 150, apertureSize=5)
    # edges, _ = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY_INV)
    
    minLineLength = grey.shape[1] // 20
    lines = cv2.HoughLinesP(image=edges,
                            rho=2,
                            theta=np.pi/2, 
                            threshold=0,
                            lines=np.array([]), 
                            minLineLength=minLineLength,
                            maxLineGap=5)

    for line in lines:
        cv2.line(line_img, 
                 (line[0][0], line[0][1]), 
                 (line[0][2], line[0][3]), 
                 (0, 255, 0), 7, 
                 cv2.LINE_AA)

    plt.subplot(1,2,1)
    plt.imshow(edges)
    plt.subplot(1,2,2)
    plt.imshow(line_img)
    plt.show()