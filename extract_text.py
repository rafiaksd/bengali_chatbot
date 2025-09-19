import os
import time
import numpy as np
from pdf2image import convert_from_path
import easyocr
from PIL import Image

reader = easyocr.Reader(['bn'])

def extract_text_from_pdf_using_ocr(pdf_path):
    pages = convert_from_path(pdf_path, 300)  # 300 DPI for better quality
    all_text = []
    
    for page_num, page_image in enumerate(pages):
        print(f"Processing page {page_num + 1}...")
        
        # Convert the PIL image to NumPy array (easyocr expects this)
        img_array = np.array(page_image)
        results = reader.readtext(img_array, detail=0)  # detail=0 means only text, not bounding boxes
        all_text.append("\n".join(results))
    
    full_text = "\n".join(all_text)
    
    return full_text

pdf_path = "bengali_pdf/bengali_small.pdf"
extracted_text = extract_text_from_pdf_using_ocr(pdf_path)

with open("extracted_bengali_text_from_images.txt", "w", encoding="utf-8") as file:
    file.write(extracted_text)

print("OCR extraction completed and saved to 'extracted_bengali_text_from_images.txt'")
