import sys
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdftypes import resolve1
from pdfminer.layout import LAParams
from pdfminer.converter import PDFPageAggregator
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.layout import *

# if True:
#     filename = 'SOKA.pdf'
def get_boxes(filename):
    fp = open(filename, 'rb')

    # parser = PDFParser(fp)
    # doc = PDFDocument(parser)
    # fields = resolve1(doc.catalog['AcroForm'])['Fields']
    
    pages = list(PDFPage.get_pages(fp))

    #Create resource manager
    rsrcmgr = PDFResourceManager()
    # Set parameters for analysis.
    laparams = LAParams(
                        # detect_vertical=True,
                        # all_texts=True,
                        # char_margin=0.02,
                        # word_margin=0.01,
                        # line_margin=0.2,
                        # line_overlap=0.0,
                        # boxes_flow=0.0 
    )
    # Create a PDF page aggregator object.
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    all_boxes = []
    for page_idx, page in enumerate(pages):
        print(page_idx)
        if (page_idx > 3):
            all_boxes.append([])
            continue
        _, _, w, h = page.mediabox
        boxes = []
        
        interpreter.process_page(page)
        # receive the LTPage object for the page.
        layout = device.get_result()
        for element in layout:
            if not isinstance(element, LTCurve):
                continue
            
            # print(element.get_text(), element.bbox)
            # text = element.get_text()
            rect = element.bbox
        
        # for i in fields[:]:
        #     field = resolve1(i)
        #     # print(field)
        #     # name, value = field.get('T'), field.get('V')
        #     # print('{0}: {1}'.format(name, value))
        #     rect = field.get('Rect')
        #     if rect is None:
        #         continue
            x1, y1, x2, y2 = rect
            y1 = (h - y1) / h
            y2 = (h - y2) / h
            x1 /= w
            x2 /= w
            box = [x1, y1, x2, y2]
            # print(box)
            boxes.append(box)
            # break
        all_boxes.append(boxes)
        
    return all_boxes

