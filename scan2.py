import pdfplumber
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import *

# infile = "2019-20 ISFAA.pdf"
infile = 'SOKA.pdf'
with pdfplumber.open(infile) as pdf:
    page = pdf.pages[2]
    _, _, pdf_w, pdf_h = page.mediabox

    img = page.to_image(resolution=300)
    
    texts = page.extract_words(
                               # x_tolerance=0, 
                               y_tolerance=0
                               )
    texts = group_boxes(texts)
    np_img = np.asarray(img.original)
    img_h, img_w = np_img.shape[:2]
    
    texts = [{'x0': int(text['x0'] / pdf_w * img_w),
              'x1': int(text['x1'] / pdf_w * img_w),
              'y0': int(text['top'] / pdf_h * img_h),
              'y1': int(text['bottom'] / pdf_h * img_h),
              'value': text['text']} 
             for text in texts]
    
    fields, remainng_texts = split_fields(texts, np_img)
    for field in fields:
        key_pos = field['key_pos']
        field_pos = field['field_pos']
        cv2.rectangle(np_img, tuple(key_pos[:2]), tuple(key_pos[2:]), (255,0,0), 5)
        cv2.rectangle(np_img, tuple(field_pos[:2]), tuple(field_pos[2:]), (0,0,255), 5)

    # img = img.draw_rects(page.rects, stroke=(0,255,0))
    
    # img = img.draw_lines(page.lines, stroke_width=4)
    
    # print(page.lines)
    
    plt.imshow(np_img)
    plt.show()
# img