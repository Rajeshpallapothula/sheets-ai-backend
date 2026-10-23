from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import json
import re
import urllib.request
import urllib.error
import base64
import pdfplumber
from io import BytesIO
from PIL import Image, ImageOps

app = FastAPI(title="Google Sheets AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str
    current_data: list
    api_key: str

def groq_request(payload_dict: dict, api_key: str) -> dict:
    if not api_key:
        raise HTTPException(status_code=401, detail="API Key missing")
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = json.dumps(payload_dict).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key.strip()}",
            "User-Agent": "Mozilla/5.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=50) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode("utf-8")
        print(f"Groq API Error: {error_msg}")
        raise HTTPException(status_code=400, detail=f"Groq Error: {error_msg}")
    except Exception as e:
        print(f"Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")

def parse_json_safely(raw_text: str) -> dict:
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {"rows": data}
    except:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match: return json.loads(match.group(0))
    return {"rows": []}

def process_vision_ocr(image_bytes: bytes, api_key: str) -> list:
    img = Image.open(BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB": img = img.convert("RGB")
    img.thumbnail((800, 800), Image.Resampling.LANCZOS)
    
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

    res = groq_request({
        "model": "qwen/qwen3.8-27b", 
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "Extract all tables, text, and handwritten data. Return ONLY a valid JSON object in this exact format: {\"rows\": [[\"Header1\", \"Header2\"], [\"Data1\", \"Data2\"]]}"
                },
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                }
            ]
        }],
        "temperature": 0.1,
        "max_tokens": 4096
    }, api_key)
    
    parsed = parse_json_safely(res["choices"][0]["message"]["content"])
    return parsed.get("rows", [])

def extract_pdf(pdf_bytes: bytes) -> list:
    path = "temp.pdf"
    with open(path, "wb") as f: f.write(pdf_bytes)
    rows = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        for row in table:
                            rows.append([str(c).strip() if c else "" for c in row])
    finally:
        if os.path.exists(path): os.remove(path)
    return rows

@app.post("/api/convert-files")
async def convert_multiple_files(files: List[UploadFile] = File(...), api_key: str = Form(...)):
    all_rows = []
    for file in files:
        raw_bytes = await file.read()
        is_pdf = file.filename.lower().endswith(".pdf")
        rows = extract_pdf(raw_bytes) if is_pdf else process_vision_ocr(raw_bytes, api_key)
        if rows:
            if all_rows and len(rows) > 1 and [str(x).upper() for x in all_rows[0]] == [str(x).upper() for x in rows[0]]:
                rows = rows[1:]
            all_rows.extend(rows)
    return {"reply": f"✅ Processed {len(files)} files.", "rows": all_rows}

@app.post("/api/ai-assistant")
async def ai_assistant_endpoint(req: PromptRequest):
    res = groq_request({
        "model": "openai/gpt-oss-120b", 
        "messages": [
            {
                "role": "system",
                "content": "You are a Google Sheets AI Copilot. Return ONLY a valid JSON object in this exact format: {\"rows\": [[\"Col1\", \"Col2\"], [\"Val1\", \"Val2\"]]}"
            },
            {"role": "user", "content": f"Current Sheet Data: {json.dumps(req.current_data[:40])}\nUser Command: {req.prompt}\nRemember to output ONLY JSON."}
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
        "response_format": {"type": "json_object"}
    }, req.api_key)
    
    parsed = parse_json_safely(res["choices"][0]["message"]["content"])
    return {"reply": "✅ Updated spreadsheet.", "rows": parsed.get("rows", [])}