from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2

from main import extract_codes_from_image

app = FastAPI(title="Strike-through OCR API")


def read_imagefile(file: UploadFile) -> np.ndarray:
    """Read an uploaded file into an OpenCV image."""
    file_bytes = np.frombuffer(file.file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Không thể đọc ảnh. Định dạng không được hỗ trợ.")
    return img


@app.post("/ocr")
async def extract_codes(file: UploadFile = File(...)):
    """
    Nhận ảnh đầu vào, trả về danh sách text 8 ký tự
    đã được OCR sau khi loại bỏ strike-through lines.
    """
    try:
        img = read_imagefile(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    matches = extract_codes_from_image(img)

    return JSONResponse({"matches": matches})

