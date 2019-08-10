import pdfplumber
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import group_boxes, detect_dash

box = '☐'
dash = '_'

# infile = "2019-20 ISFAA.pdf"
infile = 'SOKA.pdf'
with pdfplumber.open(infile) as pdf:
    page = pdf.pages[2]
    img = page.to_image(resolution=300)
    
    texts = page.extract_words(
                               # x_tolerance=0, 
                               y_tolerance=0
                               )
    texts = group_boxes(texts)
    for text in texts:
        print(text['text'])

    # img = img.draw_rects(texts, stroke=(0,255,0))
    
    # img = img.draw_lines(page.lines, stroke_width=4)
    
    tmp = np.asarray(img.annotated)
    plt.imshow(tmp)
    plt.show()
    # detect_dash(np.asarray(img.original))
# img