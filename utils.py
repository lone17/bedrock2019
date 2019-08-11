import re

import cv2
import numpy as np
from PIL import Image
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

def detect_dashes(line_img):
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

def split_fields(page, np_img):
    _, _, pdf_w, pdf_h = page.mediabox
    img_h, img_w = np_img.shape[:2]
    
    texts = page.extract_words(
                               # x_tolerance=0, 
                               y_tolerance=0
                               )
    texts = group_boxes(texts)
    
    texts = [{'x0': int(text['x0'] / pdf_w * img_w),
              'x1': int(text['x1'] / pdf_w * img_w),
              'y0': int(text['top'] / pdf_h * img_h),
              'y1': int(text['bottom'] / pdf_h * img_h),
              'value': text['text']} 
             for text in texts if text['bottom'] / pdf_h < 0.90]
    
    checkbox = r'☐'
    dash = '_'

    remaining_texts = []
    fields = []
    
    for text in texts:
        x0 = text['x0']
        x1 = text['x1']
        y0 = text['y0']
        y1 = text['y1']
        remain = True
        if dash * 5 in text['value']:
            key = text['value'].replace(dash, '').strip()
            if key != '':
                line = detect_dashes(np_img[y0:y1, x0:x1])
                field_pos = [line[0] + x0, y0, line[2] + x0, y1] 
                num_lines = (y1 - y0) // 25
                fields.append({'key': key,
                               'key_pos': [x0, y0, field_pos[0], y1],
                               'field_pos': field_pos,
                               'type': '_'.append('text', str(num_lines))})
                remain = False
            else:
                num_lines = (y1 - y0) // 25
                fields.append({'field_pos': [x0, y0, x1, y1], 
                               'type': '_'.append('text', str(num_lines))})
                remain = False
        elif bool(re.search(checkbox, text['value'])):
            # print(text['value'])
            key = re.sub(checkbox, '', text['value']).strip()
            if key != '':
                squares = detect_squares(np_img[y0:y1, x0:x1])
                if len(squares) == 1:
                    # print(squares)
                    square = squares[0]
                    checkbox_pos = [square[0] + x0, square[1] + y0, 
                                    square[2] + x0, square[3] + y0]
                    if checkbox_pos[0] - x0 > x1 - checkbox_pos[2]:
                        key_pos = [x0, y0, checkbox_pos[0], y1]
                    else:
                        key_pos = [checkbox_pos[2], y0, x1, y1]
                    fields.append({'key': key,
                                   'key_pos': key_pos,
                                   'field_pos': checkbox_pos,
                                   'type': 'checkbox'})
                    remain = False
            else:
                # print(text['value'])
                fields.append({'field_pos': [x0, y0, x1, y1],
                               'type': 'checkbox'})
                remain = False
        
        if remain:
            remaining_texts.append(text)

    return fields, remaining_texts


def detect_squares(img):
    # plt.imshow(img)
    # plt.show()
    def angle_cos(p0, p1, p2):
        d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
        return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1)*np.dot(d2, d2)))
    
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv2.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            bin, contours, _hierarchy = \
                cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours[1:]:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if (len(cnt) == 4 
                    and cv2.contourArea(cnt) > 1000 
                    and cv2.isContourConvex(cnt)):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos(cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4]) 
                                      for i in range(4)])
                    if max_cos < 0.1:
                        # squares.append(cnt)
                        (x,y,w,h) = cv2.boundingRect(cnt)
                        if [x, y, x+w, y+h] not in squares:
                            squares.append([x, y, x+w, y+h])
    
    # cv2.drawContours(img, squares, -1, (0, 255, 0), 3 )
    # print(squares)
    for box in squares:
        print(box)
        cv2.rectangle(img, tuple(box[:2]), tuple(box[2:]), (255,0,0), 1)
    plt.imshow(img)
    plt.show()
    # if len(squares) > 1:
    #     return squares[1:]
    return squares

def detect_table_cells(page, np_img):
    _, _, pdf_w, pdf_h = page.mediabox
    # print(pdf_h, pdf_w)
    img_h, img_w = np_img.shape[:2]
    
    fields = []

    table_settings = {
        'horizontal_strategy': 'lines',
        'vertical_strategy': 'lines',
        'edge_min_length': 50,
        'intersection_tolerance': 5,
        # 'text_y_tolerance':0,
        # 'text_x_tolerance': 0,
        'snap_tolerance': 3,
        'join_tolerance': 3,
    }
    for table in page.find_tables(table_settings):
        # print(table.bbox)
        _, _, table_w, table_h = table.bbox
        for cell in table.cells:
            texts = page.crop(cell).extract_words()
            # print(texts)
            if len(texts) == 0:
                # print('\t', cell)
                field_pos = [
                             int(cell[0] / pdf_w * img_w),
                             int(cell[1] / pdf_h * img_h),
                             int(cell[2] / pdf_w * img_w),
                             int(cell[3] / pdf_h * img_h),
                             ]
                # print(field_pos)
                if 3 * (cell[2] - cell[0]) > (cell[3] - cell[1]):
                    num_lines = (cell[3] - cell[1]) // 25
                    fields.append({'field_pos': field_pos, 
                                   'type': '_'.append('text', str(num_lines))})
                
    # print(len(fields))
    return fields

def format_fields(fields, np_img):
    img_h, img_w = np_img.shape[:2]
    formatted = []
    for f in fields:
        x0, y0, x1, y1 = f['field_pos']
        item = {
            'x0': x0 / img_w,
            'x1': x1 / img_w,
            'y0': y0 / img_h,
            'y1': y1 / img_h,
            'type': f['type'],
            'key': {
                   'name': f.get('key', None),
                   'pos': f.get('key_pos', None),
            }
        }
        formatted.append(item)
    
    return formatted