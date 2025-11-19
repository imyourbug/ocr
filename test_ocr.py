import cv2
import numpy as np
from main import remove_strike_lines, extract_codes_from_image
import pytesseract
import re


def test_ocr_on_image(image_path):
    """Test các hàm OCR trên ảnh input"""
    print(f"=== Testing OCR on {image_path} ===\n")

    # Load ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Lỗi: Không load được ảnh từ {image_path}")
        return

    print(f"✓ Ảnh loaded thành công")
    print(f"  Kích thước: {img.shape}\n")

    # Bước 1: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"✓ Converted to grayscale")
    print(f"  Shape: {gray.shape}\n")

    # Bước 2: Remove strike lines
    cleaned = remove_strike_lines(gray)
    print(f"✓ Removed strike lines\n")

    # Lưu ảnh sau khi xử lý (để debug)
    cv2.imwrite("image/12743_cleaned.jpg", cleaned)
    print(f"✓ Saved cleaned image to image/12743_cleaned.jpg\n")

    # Bước 3: OCR
    print("=== OCR Results ===")
    text = pytesseract.image_to_string(cleaned, config="--psm 6")
    print(f"Raw OCR text:\n{repr(text)}\n")
    print(f"Formatted OCR text:\n{text}\n")

    # Bước 4: Extract patterns
    print("=== Pattern Matching ===")

    # Pattern 1: 8 ký tự liên tiếp (không có dấu)
    matches_8char = re.findall(r"\b[A-Za-z0-9]{8}\b", text)
    print(f"8-character matches: {matches_8char}")

    # Pattern 2: 8 ký tự ngăn cách bằng dấu gạch ngang (A-B-C-D-E-F-G-H)
    matches_hyphen = re.findall(r"\b(?:[A-Za-z0-9]-){7}[A-Za-z0-9]\b", text)
    print(f"Hyphenated matches (A-B-C-D-E-F-G-H): {matches_hyphen}")
    matches_hyphen_cleaned = [h.replace("-", "") for h in matches_hyphen]
    print(f"Hyphenated cleaned: {matches_hyphen_cleaned}")

    # Tất cả matches
    all_matches = matches_8char + matches_hyphen_cleaned
    print(f"\nFinal matches: {all_matches}")

    # Bước 5: Sử dụng hàm chính
    print("\n=== extract_codes_from_image() Result ===")
    result = extract_codes_from_image(img)
    print(f"Result: {result}\n")

    return {
        "raw_text": text,
        "matches_8char": matches_8char,
        "matches_hyphen": matches_hyphen,
        "final_result": result,
    }


if __name__ == "__main__":
    # Test trên ảnh 12743.jpg
    result = test_ocr_on_image("image/12743.jpg")
