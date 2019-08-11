import pdfplumber
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import *

# infile = "2019-20 ISFAA.pdf"
pdf_file = 'SOKA.pdf'

def scan(pdf_file, page_idx):
    with pdfplumber.open(pdf_file) as pdf:
        page = pdf.pages[page_idx]

        img = page.to_image(resolution=300)
        np_img = np.asarray(img.original)
        
        fields, remaining_texts = split_fields(page, np_img)
        fields += detect_table_cells(page, np_img)
        
        # for field in fields:
        #     # print(field)
        #     key_pos = field.get('key_pos', None)
        #     field_pos = field.get('field_pos', None)
        #     if key_pos:
        #         cv2.rectangle(np_img, tuple(key_pos[:2]), tuple(key_pos[2:]), (255,0,0), 5)
        #     if field_pos:
        #         cv2.rectangle(np_img, tuple(field_pos[:2]), tuple(field_pos[2:]), (0,0,255), 5)

        # cells = []
        # table_settings = {
        #     'horizontal_strategy': 'lines',
        #     'vertical_strategy': 'lines',
        #     'edge_min_length': 100,
        # }
        # for table in page.find_tables(table_settings):
        #     cells += table.cells
        # img = img.draw_rects(cells, stroke=(0,255,0), stroke_width=5)
        # img = img.draw_rects(texts, stroke=(0,255,0), stroke_width=11)
        
        # img = img.draw_lines(page.lines, stroke_width=4)
        
        # print(page.lines)
        
        # plt.imshow(np_img)
        # plt.show()
        
        fields = format_fields(fields, np_img)
        
        return fields

if __name__ == '__main__':
    for f in scan(pdf_file, 5):
        print(f)