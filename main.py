from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import base64
from fastapi import FastAPI, Request
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

app = FastAPI()

# Point to the templates folder
templates = Jinja2Templates(directory="templates")

# Serve static files (if needed for CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# CORS for frontend-backend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  #Ideally, use your frontend actual origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini model setup
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)

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
