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

# High-speed model priority chain
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
    return {"status": "online", "message": "Server is awake and ready!"}

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
            "max_tokens": 2048
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
            with urllib.request.urlopen(req, timeout=45) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_msg = e.read().decode("utf-8")
            last_error = f"HTTP {e.code}: {error_msg}"

            if e.code == 401 or "invalid_api_key" in error_msg.lower():
                raise HTTPException(
                    status_code=401,
                    detail="Unauthorized API Key. Please generate a fresh key on Groq Console and click Save."
                )
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
            return {"reply": "✅ Created your template!", "action": "DIRECT_TEMPLATE", "rows": data}
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
    img.thumbnail((800, 800), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=75)
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
        "You are a fast, human-aware Google Sheets AI Copilot and Universal Template Architect.\n"
        "Instead of offering fixed templates, you dynamically design custom, official-style Google Sheets templates for ANY user intention (timetables, study planners, business trackers, budgets, workouts, inventories, event schedules, logs, or OCR extractions).\n"
        "You also understand the sheet like a human—by its Title Banner, Instructions Box, Subheadings Row, and Data Rows—so the user never needs to mention cell coordinates like A1.\n\n"
        "Return ONLY a raw valid JSON object in this exact structure:\n"
        "{\n"
        "  \"reply\": \"Brief, friendly confirmation of what you built or updated\",\n"
        "  \"action\": \"DIRECT_TEMPLATE | UPDATE_SUBHEADINGS | UPDATE_TITLE | OPEN_TEMPLATE_GALLERY | CLEAR_SHEET | NEW_FILE | NEW_TAB | FORMAT_TABLE | FREEZE_HEADER | SORT_ASC | SORT_DESC | REMOVE_DUPLICATES\",\n"
        "  \"target_name\": \"Short clean tab or file name\",\n"
        "  \"template_title\": \"LARGE BANNER TITLE MATCHING USER INTENT\",\n"
        "  \"template_instructions\": \"1-2 helpful sentences explaining how to use this custom sheet\",\n"
        "  \"new_subheadings\": [\"Col1\", \"Col2\"],\n"
        "  \"dropdowns\": {\"ColumnName\": [\"Option1\", \"Option2\", \"Option3\"]},\n"
        "  \"rows\": [[\"Subheading1\", \"Subheading2\"], [\"Sample1\", \"Sample2\"]]\n"
        "}\n\n"
        "CRITICAL RULES:\n"
        "1. DIRECT OFFICIAL TEMPLATE BY DEFAULT: Whenever the user asks to create, generate, track, schedule, plan, organize, or modify a table/timetable/template/sheet (or extracts from a photo/PDF), ALWAYS use action=\"DIRECT_TEMPLATE\". Never output an unstyled plain table first.\n"
        "2. TAILOR TO USER INTENTION: Design 'rows[0]' (the subheadings), 'rows[1..]' (3 to 5 realistic starter rows), 'template_title', 'template_instructions', and 'dropdowns' (for any status, category, priority, day, or type column) specifically around the user's goal.\n"
        "3. IN-PLACE HUMAN EDITS: If a template is already on the sheet (check 'landmarks.table_rows') and the user asks to edit a cell or row (e.g., 'put English in Tuesday period 1', 'add a row for John'), keep the existing template structure, apply the exact edit to 'rows', preserve 'template_title' and 'template_instructions', and return action=\"DIRECT_TEMPLATE\".\n"
        "4. SUBHEADING EDITS: If the user asks to change, rename, or add a word to the subheadings of the table just created, set action=\"UPDATE_SUBHEADINGS\" and provide the full updated header list in 'new_subheadings'.\n"
        "5. OFFICIAL TEMPLATE GALLERY: Only if the user explicitly says 'go to template gallery' or 'open official google template gallery', set action=\"OPEN_TEMPLATE_GALLERY\" and rows=[].\n"
        "6. Output raw JSON only, with no markdown code fences."
    )

    user_instruction = prompt if prompt else "Extract all data into a clean, structured official template matching the content."
    content_blocks = [
        {
            "type": "text",
            "text": (
                f"{system_instruction}\n\n"
                f"Recent Memory: {json.dumps(history_list[-4:])}\n"
                f"Current Sheet Landmarks: {json.dumps(sheet_context.get('landmarks', {}))}\n"
                f"User Request: {user_instruction}"
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
                "text": f"Attached PDF ({file.filename}):\n{pdf_content[:5000]}"
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
        "reply": parsed.get("reply", "✅ Template ready!"),
        "action": parsed.get("action", "DIRECT_TEMPLATE"),
        "target_name": parsed.get("target_name", ""),
        "template_title": parsed.get("template_title", ""),
        "template_instructions": parsed.get("template_instructions", ""),
        "new_subheadings": parsed.get("new_subheadings", []),
        "dropdowns": parsed.get("dropdowns", {}),
        "rows": parsed.get("rows", [])
    }