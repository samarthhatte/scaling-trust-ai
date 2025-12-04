from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

import os
import base64
import httpx
import pdfplumber
from docx import Document

# Google GenAI SDK (NEW)
from google import genai

os.environ["GOOGLE_API_KEY"] = "AIzaSyBXTuOEK6RxsCu6RHWf9hE1hfGtZXb0UcU"
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

origins = [
    "https://hackers.kesug.com",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ======================================
# HOME PAGE
# ======================================
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ======================================
# IMAGE ANALYSIS (FIXED)
# ======================================
@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    image_bytes = await file.read()

    reply = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            {
                "parts": [
                    {"text": "Categorize this image as Harmful, Neutral, or Good."},
                    {"inline_data": {"mime_type": file.content_type, "data": base64.b64encode(image_bytes).decode()}}
                ]
            }
        ]
    )

    return {"category": reply.text}


# ======================================
# TEXT ONLY
# ======================================
@app.post("/api/ask")
async def ask_gemini(request: Request):
    body = await request.json()
    prompt = body.get("prompt")

    if not prompt:
        return {"message": "Prompt is missing."}

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"You are a helpful healthcare assistant.\n\n{prompt}"
    )

    return {"message": response.text}


# ======================================
# IMAGE + QUESTION
# ======================================
@app.post("/api/ask-with-image")
async def ask_with_image(file: UploadFile = File(...), prompt: str = Form(...)):
    image_bytes = await file.read()

    reply = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            {
                "parts": [
                    {"text": f"Analyze this image and answer:\n{prompt}"},
                    {"inline_data": {"mime_type": file.content_type, "data": base64.b64encode(image_bytes).decode()}}
                ]
            }
        ]
    )

    return {"response": reply.text}


# ======================================
# DOC + QUESTION
# ======================================
@app.post("/api/ask-with-doc")
async def ask_with_doc(file: UploadFile = File(...), prompt: str = Form(...)):
    extension = file.filename.lower().split('.')[-1]
    file_bytes = await file.read()
    extracted = ""

    if extension == "txt":
        extracted = file_bytes.decode("utf-8", errors="ignore")

    elif extension == "pdf":
        with open("temp.pdf", "wb") as f:
            f.write(file_bytes)
        with pdfplumber.open("temp.pdf") as pdf:
            extracted = "\n".join((p.extract_text() or "") for p in pdf.pages)
        os.remove("temp.pdf")

    elif extension in ["doc", "docx"]:
        with open("temp.docx", "wb") as f:
            f.write(file_bytes)
        doc = Document("temp.docx")
        extracted = "\n".join(p.text for p in doc.paragraphs)
        os.remove("temp.docx")

    else:
        return {"message": f"Unsupported file type: {extension}"}

    content = f"You are a helpful healthcare assistant.\n\nQuestion: {prompt}\n\nDocument:\n{extracted}"

    reply = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=content
    )

    return {"message": reply.text}


# ======================================
# MENTAL HEALTH SUPPORT CHAT
# ======================================
@app.post("/api/mental-support-chat")
async def mental_support_chat(request: Request):
    body = await request.json()
    user_message = body.get("message")

    if not user_message:
        return {"message": "Message is required"}

    system_prompt = """
    You are a supportive companion for emotional well-being conversations.
    You must ALWAYS follow these rules:

    1. If the user expresses suicidal thoughts, self-harm intent, or mentions harming others:
        - DO NOT provide therapy, coping exercises, grounding techniques, or step-by-step advice.
        - DO NOT attempt to persuade or talk them out of it.
        - DO NOT ask them to describe their plan or intent.
        - DO NOT say you understand exactly how they feel.
        - DO NOT give medical or legal advice.

    You MUST respond in this format:

    A) Acknowledge their feelings with warmth and non-judgement.
    B) Encourage them to reach out to someone they trust right now.
    C) Provide emergency contact suggestions:
       - If they are in India: Tell them to call Arogya Vani 104 or National Suicide Prevention Helpline India at 9152987821.
       - Or local emergency number (112).
    D) Encourage contacting a close friend, family member, or mental health professional immediately.
    E) Remind them they are not alone.

    2. For normal emotional concerns:
        - Use gentle supportive language.
        - Reflective listening.
        - No diagnosis.
        - No medical claims.

    You must NOT mention being an AI model.
    """

    final_prompt = f"{system_prompt}\nUser: {user_message}\nAssistant:"

    reply = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=final_prompt
    )

    return {"reply": reply.text}
