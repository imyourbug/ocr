from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import pytesseract
import re
import requests
import json
import base64
import os
from dotenv import load_dotenv

load_dotenv()

from main import extract_codes_from_image, remove_strike_lines

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

    return JSONResponse({"text": matches})


@app.post("/ocr-with-prompt")
async def extract_text_with_prompt(
    file: UploadFile = File(...), prompt: str = Form(...)
):
    """
    Nhận ảnh và prompt đầu vào, thực hiện OCR và extract text theo prompt.

    Ví dụ prompts:
    - "extract_8_chars": Trích xuất 8 ký tự liên tiếp
    - "extract_hyphenated": Trích xuất 8 ký tự ngăn cách bằng dấu gạch ngang (A-B-C-D-E-F-G-H)
    - "extract_all": Trích xuất tất cả (8 ký tự + hyphenated)
    - "raw_text": Trả về OCR text thô
    - "clean_text": Trả về OCR text sau khi loại bỏ strike-through
    """
    try:
        img = read_imagefile(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Remove strike lines
        cleaned = remove_strike_lines(gray)

        # OCR
        text = pytesseract.image_to_string(cleaned, config="--psm 6")

        # Process based on prompt
        if prompt == "raw_text":
            return JSONResponse({"text": [text]})

        elif prompt == "clean_text":
            # Return cleaned image OCR result
            return JSONResponse({"text": [text]})

        elif prompt == "extract_8_chars":
            # Extract 8 consecutive characters
            matches = re.findall(r"\b[A-Za-z0-9]{8}\b", text)
            return JSONResponse({"text": matches})

        elif prompt == "extract_hyphenated":
            # Extract hyphenated format (A-B-C-D-E-F-G-H)
            matches = re.findall(r"\b(?:[A-Za-z0-9]-){7}[A-Za-z0-9]\b", text)
            matches_cleaned = [m.replace("-", "") for m in matches]
            return JSONResponse({"text": matches_cleaned})

        elif prompt == "extract_all":
            # Extract both formats
            matches_8char = re.findall(r"\b[A-Za-z0-9]{8}\b", text)
            matches_hyphen = re.findall(r"\b(?:[A-Za-z0-9]-){7}[A-Za-z0-9]\b", text)
            matches_hyphen_cleaned = [m.replace("-", "") for m in matches_hyphen]
            all_matches = matches_8char + matches_hyphen_cleaned

            return JSONResponse({"text": all_matches})

        else:
            raise ValueError(
                f"Unknown prompt: {prompt}. Valid prompts: raw_text, clean_text, extract_8_chars, extract_hyphenated, extract_all"
            )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def encode_image_to_base64(image_bytes: bytes) -> str:
    """Encode image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")


@app.post("/ocr-with-llm")
async def extract_text_with_llm(
    file: UploadFile = File(...),
    model: str = Form(default="openrouter/sherlock-dash-alpha"),
):
    """
    Sử dụng LLM (qua OpenRouter API) để extract text từ ảnh.

    Parameters:
    - file: Ảnh cần xử lý
    - prompt: Hướng dẫn cho LLM (VD: "Extract all 8-character codes from this image")
    - model: Model LLM (default: google/gemini-2.0-flash-001)

    Cần set env variable: OPENROUTER_API_KEY
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500, detail="OPENROUTER_API_KEY environment variable not set"
        )

    try:
        # Read image
        file_bytes = await file.read()
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Không thể đọc ảnh. Định dạng không được hỗ trợ.")

        # Encode image to base64
        base64_image = encode_image_to_base64(file_bytes)
        data_url = f"data:image/jpeg;base64,{base64_image}"

        # Prepare request to OpenRouter
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        prompt = """Extract all texts from the image. Return only all texts you get in a JSON array.
        Example response:
        ['text1', 'text2', 'text3', ...,'textN']
        """

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]

        payload = {"model": model, "messages": messages}

        # Call OpenRouter API
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        result = response.json()

        # Extract the text from LLM response
        llm_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Parse LLM response as JSON array if possible, otherwise wrap in array
        try:
            parsed_text = json.loads(llm_text)
            if isinstance(parsed_text, list):
                text_list = parsed_text
            else:
                text_list = [llm_text]
        except (json.JSONDecodeError, TypeError):
            text_list = [llm_text]

        return JSONResponse({"text": text_list})

    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=500, detail=f"OpenRouter API error: {str(exc)}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
