import cv2
import numpy as np
import pytesseract
import re

from typing import List

# Cập nhật đường dẫn nếu cần
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def remove_strike_lines(img_gray):
    # Nhị phân hóa
    th = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV,21,10)
    
    # Phát hiện và xóa đường ngang
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
    detected_lines = cv2.morphologyEx(th, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    
    contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.ones_like(img_gray) * 255
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(mask, (x-2, y-2), (x+w+2, y+h+2), 0, -1)

    cleaned = cv2.bitwise_and(img_gray, img_gray, mask=mask)
    return cv2.medianBlur(cleaned, 3)

def extract_codes_from_image(img: np.ndarray) -> List[str]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cleaned = remove_strike_lines(gray)

    text = pytesseract.image_to_string(cleaned, config='--psm 6')
    matches = re.findall(r'\b[A-Za-z0-9]{8}\b', text)
    hyph = re.findall(r'\b(?:[A-Za-z0-9]-){7}[A-Za-z0-9]\b', text)
    matches += [h.replace('-', '') for h in hyph]
    return matches


def ocr_extract(image_path):
    img = cv2.imread(image_path)
    return extract_codes_from_image(img)

if __name__ == "__main__":
    results = ocr_extract("sample_strikethrough.jpg")
    print("Code 8 ký tự tìm thấy:", results)
