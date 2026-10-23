from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
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

TEXT_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
    "llama-3.3-70b-versatile"
]

VISION_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct"
]

def groq_request_with_fallback(messages: list, api_key: str, is_vision: bool = False) -> dict:
    if not api_key or not api_key.strip():
        raise HTTPException(status_code=401, detail="API Key missing.")
    
    models_to_try = VISION_MODELS if is_vision else TEXT_MODELS
    last_error = ""

    for model_name in models_to_try:
        payload_dict = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 4096
        }
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
            with urllib.request.urlopen(req, timeout=55) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_msg = e.read().decode("utf-8")
            last_error = f"HTTP {e.code}: {error_msg}"
            if "model_not_found" in error_msg or "decommissioned" in error_msg or e.code == 404:
                continue
            raise HTTPException(status_code=400, detail=f"Groq Error: {error_msg}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")

    raise HTTPException(status_code=400, detail=f"Groq Model Error: {last_error}")

def parse_json_safely(raw_text: str) -> dict:
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            return {"reply": "✅ Updated spreadsheet.", "action": "UPDATE_DATA", "rows": data}
    except:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    return {"reply": raw_text, "action": "NONE", "rows": []}

def image_to_base64(image_bytes: bytes) -> str:
    img = Image.open(BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail((900, 900), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def extract_pdf_text_and_tables(pdf_bytes: bytes) -> str:
    path = "temp.pdf"
    with open(path, "wb") as f:
        f.write(pdf_bytes)
    extracted = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        for row in table:
                            extracted.append(str([str(c).strip() if c else "" for c in row]))
                else:
                    text = page.extract_text()
                    if text:
                        extracted.append(text)
    finally:
        if os.path.exists(path):
            os.remove(path)
    return "\n".join(extracted)

@app.post("/api/copilot")
async def unified_copilot(
    api_key: str = Form(...),
    prompt: str = Form(""),
    current_data: str = Form("[]"),
    files: Optional[List[UploadFile]] = File(default=None)
):
    try:
        sheet_rows = json.loads(current_data) if current_data else []
    except:
        sheet_rows = []

    system_instruction = (
        "You are an intelligent Google Sheets AI Copilot designed for beginners. "
        "You can both edit data AND trigger Google Sheets tools automatically.\n"
        "You MUST return ONLY a valid JSON object in this exact format:\n"
        "{\n"
        "  \"reply\": \"Friendly explanation of what you did\",\n"
        "  \"action\": \"UPDATE_DATA\",\n"
        "  \"target_name\": \"Optional name for template, tab, or new file\",\n"
        "  \"rows\": [[\"Header1\", \"Header2\"], [\"Row1Col1\", \"Row1Col2\"]]\n"
        "}\n\n"
        "Choose the best 'action' from this exact list:\n"
        "- \"CLEAR_SHEET\": When user says 'clear sheet', 'erase everything', 'reset', or 'delete all'. Set rows to [].\n"
        "- \"OPEN_TEMPLATE\": When user asks to open or create any template (e.g., 'open attendance template', 'open invoice template', 'open budget tracker'). Set 'target_name' to the template name and populate 'rows' with a complete, realistic template including headers and sample rows.\n"
        "- \"NEW_FILE\": When user says 'open new file', 'create new spreadsheet', or 'new file'. Set 'target_name' to a title. If they also want a template inside it, populate 'rows', otherwise [].\n"
        "- \"NEW_TAB\": When user asks to open or add a new sheet/tab inside the current file. Set 'target_name' to the tab name.\n"
        "- \"FORMAT_TABLE\": When user asks to format, style, beautify, or auto-fit the table.\n"
        "- \"FREEZE_HEADER\": When user asks to freeze the top row or header.\n"
        "- \"SORT_ASC\": When user asks to sort A-Z or ascending.\n"
        "- \"SORT_DESC\": When user asks to sort Z-A or descending.\n"
        "- \"REMOVE_DUPLICATES\": When user asks to remove duplicate rows.\n"
        "- \"UPDATE_DATA\": Default action when creating, modifying, or extracting data into the current sheet.\n\n"
        "Output raw JSON only, with no markdown code blocks."
    )

    user_instruction = prompt if prompt else "Extract all data and tables cleanly into spreadsheet rows."
    content_blocks = [
        {
            "type": "text",
            "text": f"{system_instruction}\n\nCurrent Sheet Data (first 40 rows): {json.dumps(sheet_rows[:40])}\n\nUser Command: {user_instruction}"
        }
    ]

    has_images = False
    if files:
        for file in files:
            if not file.filename:
                continue
            raw_bytes = await file.read()
            if file.filename.lower().endswith(".pdf"):
                pdf_content = extract_pdf_text_and_tables(raw_bytes)
                content_blocks.append({
                    "type": "text",
                    "text": f"Extracted PDF Data ({file.filename}):\n{pdf_content[:6000]}"
                })
            else:
                b64_img = image_to_base64(raw_bytes)
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                })
                has_images = True

    res = groq_request_with_fallback(
        messages=[{"role": "user", "content": content_blocks}],
        api_key=api_key,
        is_vision=has_images
    )

    parsed = parse_json_safely(res["choices"][0]["message"]["content"])
    return {
        "reply": parsed.get("reply", "✅ Done!"),
        "action": parsed.get("action", "UPDATE_DATA"),
        "target_name": parsed.get("target_name", ""),
        "rows": parsed.get("rows", [])
    }