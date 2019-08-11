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
        'text': ' '.join([b['text'] for b in boxes]).strip()
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
    # plt.imshow(line_img)
    # plt.show()
    if len(line_img.shape) > 2:
        grey = cv2.cvtColor(line_img,cv2.COLOR_BGR2GRAY)
    else:
        grey = line_img

    # grey = cv2.bilateralFilter(grey, 9, 75, 75)
    # grey = cv2.morphologyEx(grey, cv2.MORPH_OPEN, (7,1))
    # grey = cv2.dilate(grey, (7,1), iterations=1)
    # grey = cv2.erode(grey, (7,1), iterations=1)
    edges = cv2.Canny(grey, 120, 150, apertureSize=5)
    # edges, _ = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY_INV)
    
    minLineLength = grey.shape[1] // 5
    lines = cv2.HoughLinesP(image=edges,
                            rho=5,
                            theta=np.pi/2, 
                            threshold=0,
                            lines=np.array([]), 
                            minLineLength=minLineLength,
                            maxLineGap=5)
    
    # if lines is None:
    #     return None

    lines = [line[0] for line in lines]
    
    # for line in lines:
    #     cv2.line(line_img, 
    #              (line[0], line[1]), 
    #              (line[2], line[3]), 
    #              (0, 255, 0), 7, 
    #              cv2.LINE_AA)
    
    # if lines is None:
    #     plt.subplot(1,2,1)
    #     plt.imshow(edges)
    #     plt.subplot(1,2,2)
    #     plt.imshow(line_img)
    #     plt.show()

    return max(lines, key=lambda l: l[2] - l[0])

def split_fields(texts, np_img):
    box = '☐'
    dash = '_'

    remainng_texts = []
    fields = []
    
    for text in texts:
        x0 = text['x0']
        x1 = text['x1']
        y0 = text['y0']
        y1 = text['y1']
        if dash * 5 in text['value']:
            key = text['value'].replace('_', '').strip()
            if key != '':
                line = detect_dash(np_img[y0:y1, x0:x1])
                field_pos = [line[0] + x0, y0, line[2] + x0, y1] 
                fields.append({'key': key,
                               'key_pos': [x0, y0, field_pos[0], y1],
                               'field_pos': field_pos,
                               'type': 'string'})
            else:
                remainng_texts.append(text)
        else:
            remainng_texts.append(text)
            
    return fields, remainng_texts


def detect_squares(img):
    
    def angle_cos(p0, p1, p2):
        d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
        return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )
    
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv2.split(img):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            bin, contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    
    cv.drawContours(img, squares, -1, (0, 255, 0), 3 )
    plt.imshow(img)
    plt.show()
    return squares