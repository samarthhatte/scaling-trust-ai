from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import base64
import pdfplumber
from docx import Document
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from fastapi import Request
from fastapi import UploadFile, File, Form
import httpx
import os
from fastapi import APIRouter, UploadFile, File, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import whisper
import tempfile
import uuid
import os
import pyttsx3
from pydub import AudioSegment
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
os.environ["GOOGLE_API_KEY"] = "AIzaSyBXTuOEK6RxsCu6RHWf9hE1hfGtZXb0UcU"

app = FastAPI()

# Point to the templates folder
templates = Jinja2Templates(directory="templates")

# Serve static files (if needed for CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Allow your frontend domain
origins = [
    "https://hackers.kesug.com",  # your frontend domain
    "http://localhost:3000",      # optional for local testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,             # domains allowed accessing the backend
    allow_credentials=True,
    allow_methods=["*"],               # allow all HTTP methods
    allow_headers=["*"],               # allow all headers
)
# Gemini model setup
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

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

@app.post("/api/ask")
async def ask_gemini(request: Request):
    body = await request.json()
    prompt = body.get("prompt")

    if not prompt:
        return {"message": "Prompt is missing."}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={os.getenv('GOOGLE_API_KEY')}"

    payload = {
        "contents": [
            {
                "parts": [{"text": f"You are a helpful healthcare assistant. Answer the following healthcare-related question:\n\n{prompt}"}]
            }
        ]
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        data = response.json()

    message = (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "No response from Gemini")
    )

    return {"message": message}


@app.post("/api/ask-with-image")
async def ask_with_image(
    file: UploadFile = File(...),
    prompt: str = Form(...)
):
    image_bytes = await file.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    # Create a Gemini message with both image and prompt
    combined_prompt = HumanMessage(
        content=[
            {"type": "text", "text": f"You are a helpful healthcare assistant. Analyze the following image and answer the question based on it:\n\nQuestion: {prompt}"},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}
        ]
    )

    response = llm.invoke([combined_prompt])
    answer = response.content

    return {"response": answer}


@app.post("/api/ask-with-doc")
async def ask_with_doc(
        file: UploadFile = File(...),
        prompt: str = Form(...)
):
    extension = file.filename.lower().split('.')[-1]
    file_bytes = await file.read()

    # Extract text from the uploaded document
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
        extracted_text = "\n".join([para.text for para in doc.paragraphs])
        os.remove("temp.docx")

    else:
        return {"message": f"Unsupported file type: {extension}"}

    full_prompt =  f"You are a helpful healthcare assistant. Answer the following question based on the provided document:\n\n{prompt}\n\n[Document Text]:\n{extracted_text}"

    # Gemini API call
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={os.getenv('GOOGLE_API_KEY')}"
    payload = {
        "contents": [
            {
                "parts": [{"text": full_prompt}]
            }
        ]
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        data = response.json()

    message = (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "No response from Gemini")
    )

    return {"message": message}
