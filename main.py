from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

import base64
import httpx
import os
import pdfplumber
from docx import Document

# LangChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Google API Client
from google import genai

os.environ["GOOGLE_API_KEY"] = "AIzaSyBXTuOEK6RxsCu6RHWf9hE1hfGtZXb0UcU"
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

origins = [
    "https://hackers.kesug.com",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LangChain model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)


@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ================================
#   IMAGE ANALYSIS
# ================================
@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = HumanMessage(
        content=[
            {"type": "text", "text": "Categorize this image as 'Harmful', 'Neutral', or 'Good'."},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}
        ]
    )

    response = llm.invoke([prompt])
    return {"category": response.content}

# ================================
#   TEXT ONLY
# ================================
@app.post("/api/ask")
async def ask_gemini(request: Request):
    body = await request.json()
    prompt = body.get("prompt")

    if not prompt:
        return {"message": "Prompt is missing."}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={os.getenv('GOOGLE_API_KEY')}"
    payload = {
        "contents": [
            {
                "parts": [{"text": f"You are a helpful healthcare assistant.\n\n{prompt}"}]
            }
        ]
    }

    async with httpx.AsyncClient() as client_http:
        response = await client_http.post(url, json=payload)
        data = response.json()

    message = (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "No response from Gemini")
    )

    return {"message": message}

# ================================
#   IMAGE + QUESTION
# ================================
@app.post("/api/ask-with-image")
async def ask_with_image(
    file: UploadFile = File(...),
    prompt: str = Form(...)
):
    image_bytes = await file.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    combined_prompt = HumanMessage(
        content=[
            {"type": "text", "text": f"Analyze this medical image and answer:\n\n{prompt}"},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}
        ]
    )

    response = llm.invoke([combined_prompt])
    return {"response": response.content}

# ================================
#   DOC + QUESTION
# ================================
@app.post("/api/ask-with-doc")
async def ask_with_doc(
    file: UploadFile = File(...),
    prompt: str = Form(...)
):
    extension = file.filename.lower().split('.')[-1]
    file_bytes = await file.read()
    extracted_text = ""

    if extension == "txt":
        extracted_text = file_bytes.decode("utf-8", errors="ignore")

    elif extension == "pdf":
        with open("temp.pdf", "wb") as f:
            f.write(file_bytes)
        with pdfplumber.open("temp.pdf") as pdf:
            extracted_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        os.remove("temp.pdf")

    elif extension in ["doc", "docx"]:
        with open("temp.docx", "wb") as f:
            f.write(file_bytes)
        doc = Document("temp.docx")
        extracted_text = "\n".join(p.text for p in doc.paragraphs)
        os.remove("temp.docx")

    else:
        return {"message": f"Unsupported file type: {extension}"}

    full_prompt = (
        f"You are a helpful healthcare assistant.\n\nQuestion: {prompt}\n\n"
        f"Document:\n{extracted_text}"
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={os.getenv('GOOGLE_API_KEY')}"
    payload = {"contents": [{"parts": [{"text": full_prompt}]}]}

    async with httpx.AsyncClient() as client_http:
        response = await client_http.post(url, json=payload)
        data = response.json()

    message = (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "No response from Gemini")
    )
    return {"message": message}

# ================================
#   MENTAL SUPPORT CHAT
# ================================
@app.post("/api/mental-support-chat")
async def mental_support_chat(request: Request):
    data = await request.json()
    user_message = data.get("query", "")

    if not user_message.strip():
        return {"response": "I'm here with you. Could you share what's on your mind?"}

    prompt = [
        {"role": "system", "content": """
You are a warm, empathetic mental wellness support companion.
Your tone is gentle, comforting, and human-like.
Avoid sounding robotic. Do not diagnose.
"""},
        {"role": "user", "content": user_message}
    ]

    reply = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    return {"response": reply.text.strip()}
