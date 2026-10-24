from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

app = FastAPI(title="Google Sheets AI Copilot Master Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Automatic fallback lists so retired or rate-limited models never break your app
TEXT_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b"
]

VISION_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.2-11b-vision-preview",
    "llama-3.2-90b-vision-preview"
]

@app.get("/")
async def health_check():
    return {"status": "online", "message": "Google Sheets AI Copilot Backend is Live!"}

def groq_request_with_fallback(messages: list, api_key: str, is_vision: bool = False) -> dict:
    clean_key = (api_key or "").strip()
    if not clean_key:
        raise HTTPException(status_code=401, detail="API Key is missing. Please paste and save your Groq API Key.")

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
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {clean_key}",
                "User-Agent": "Mozilla/5.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=55) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_msg = e.read().decode("utf-8")
            last_error = f"HTTP {e.code}: {error_msg}"
            print(f"Model {model_name} failed with {last_error}")

            # Immediately stop if the API key itself is invalid/unauthorized
            if e.code == 401 or "invalid_api_key" in error_msg.lower():
                raise HTTPException(
                    status_code=401,
                    detail="Unauthorized API Key. Please generate a fresh key on Groq Console and click Save."
                )
            # If model is not found, decommissioned, or rate-limited, automatically try the next model
            if e.code in (400, 404, 429) or "model" in error_msg.lower() or "rate_limit" in error_msg.lower():
                continue
            raise HTTPException(status_code=400, detail=f"Groq Error: {error_msg}")
        except Exception as e:
            last_error = str(e)
            continue

    raise HTTPException(status_code=400, detail=f"All AI models failed. Last error: {last_error}")

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
    # Downscale to 850x850 so base64 payload never triggers a 400 Bad Request
    img.thumbnail((850, 850), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def extract_pdf_text_and_tables(pdf_bytes: bytes) -> str:
    path = "temp_upload.pdf"
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
    current_data: str = Form("{}"),
    chat_history: str = Form("[]"),
    files: List[UploadFile] = File(default=[])
):
    try:
        sheet_context = json.loads(current_data) if current_data else {}
    except:
        sheet_context = {}

    try:
        history_list = json.loads(chat_history) if chat_history else []
    except:
        history_list = []

    system_instruction = (
        "You are a human-aware Google Sheets AI Copilot and Official Template Designer for beginners.\n"
        "You understand the spreadsheet like a human—by its Title Banner, Instructions Box, Subheadings Row, and Data Rows—so the user never needs to mention cell coordinates like A1 or B4.\n"
        "You also remember what you just created or edited from the Recent Conversation Memory.\n\n"
        "You MUST return ONLY a raw valid JSON object in this exact structure:\n"
        "{\n"
        "  \"reply\": \"Friendly explanation of what you did\",\n"
        "  \"action\": \"OPEN_TEMPLATE | UPDATE_SUBHEADINGS | UPDATE_TITLE | UPDATE_DATA | OPEN_TEMPLATE_GALLERY | CLEAR_SHEET | NEW_FILE | NEW_TAB | FORMAT_TABLE | FREEZE_HEADER | SORT_ASC | SORT_DESC | REMOVE_DUPLICATES\",\n"
        "  \"target_name\": \"Optional tab or file name\",\n"
        "  \"template_title\": \"LARGE BANNER TITLE (e.g., ATTENDANCE TRACKER)\",\n"
        "  \"template_instructions\": \"Short helper instructions for the top-right box\",\n"
        "  \"new_subheadings\": [\"Col1\", \"Col2\"],\n"
        "  \"dropdowns\": {\"Status\": [\"Present\", \"Absent\", \"Late\"]},\n"
        "  \"preserve_template_style\": true,\n"
        "  \"rows\": [[\"Subheading1\", \"Subheading2\"], [\"Row1Val1\", \"Row1Val2\"]]\n"
        "}\n\n"
        "CRITICAL ROUTING RULES:\n"
        "1. OFFICIAL TEMPLATE GALLERY: If the user says 'go to template gallery', 'open template gallery', or 'browse official templates', set action=\"OPEN_TEMPLATE_GALLERY\" and rows=[]. Do NOT generate a table of template names.\n"
        "2. OFFICIAL TEMPLATES (e.g., 'attendance tracker', 'assignment tracker', 'budget', 'invoice', 'project tracker'): Set action=\"OPEN_TEMPLATE\", set 'target_name' to a clean tab name, set 'template_title', write helpful 'template_instructions', define 'dropdowns' for status/category/priority columns, and provide realistic sample 'rows' where rows[0] is the subheadings array and rows[1..] are sample data rows.\n"
        "3. HUMAN MEMORY & SUBHEADING EDITS: If the user refers to the table just created (e.g., 'add a word to the subheadings', 'change the subheadings of the table we just made', 'rename the second column'), check 'landmarks.current_subheadings' and 'Recent Conversation Memory'. Set action=\"UPDATE_SUBHEADINGS\" and put the full updated list of column headers into 'new_subheadings'.\n"
        "4. HUMAN CELL/ROW EDITS (e.g., 'put English in Tuesday period 1', 'mark Bob Smith as Present', 'add 3 more rows'): Read the existing table from 'table_rows' in Current Sheet Landmarks, apply the exact change requested while keeping all other existing rows intact, and return the complete updated table in 'rows' with action=\"UPDATE_DATA\".\n"
        "5. SHEET TOOLS: Use CLEAR_SHEET, NEW_FILE, NEW_TAB, FORMAT_TABLE, FREEZE_HEADER, SORT_ASC, SORT_DESC, or REMOVE_DUPLICATES when the user asks for those tools.\n"
        "6. Output raw JSON only, without markdown code fences."
    )

    user_instruction = prompt if prompt else "Extract all tables, text, and handwritten data cleanly into structured spreadsheet rows."
    content_blocks = [
        {
            "type": "text",
            "text": (
                f"{system_instruction}\n\n"
                f"Recent Conversation Memory: {json.dumps(history_list)}\n\n"
                f"Current Sheet Landmarks & Data: {json.dumps(sheet_context)}\n\n"
                f"User Command: {user_instruction}"
            )
        }
    ]

    has_images = False
    for file in files:
        if not file or not file.filename:
            continue
        raw_bytes = await file.read()
        if not raw_bytes:
            continue
        if file.filename.lower().endswith(".pdf"):
            pdf_content = extract_pdf_text_and_tables(raw_bytes)
            content_blocks.append({
                "type": "text",
                "text": f"Attached PDF Content ({file.filename}):\n{pdf_content[:6000]}"
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
        "template_title": parsed.get("template_title", ""),
        "template_instructions": parsed.get("template_instructions", ""),
        "new_subheadings": parsed.get("new_subheadings", []),
        "dropdowns": parsed.get("dropdowns", {}),
        "preserve_template_style": parsed.get("preserve_template_style", False),
        "rows": parsed.get("rows", [])
    }