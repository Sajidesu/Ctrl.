from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time
import json
import os
from dotenv import load_dotenv
import chromadb
from PyPDF2 import PdfReader
from openai import OpenAI

# ==========================================
# 1. INITIALIZATION & INFRASTRUCTURE
# ==========================================

# Load hidden environment variables securely
load_dotenv()
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")

if not FIREWORKS_API_KEY:
    raise ValueError("CRITICAL ERROR: FIREWORKS_API_KEY is missing from the .env file!")

# Initialize Fireworks Client via OpenAI SDK
client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=FIREWORKS_API_KEY
)

# Initialize embedded local vector database for fast OS-level interception
chroma_client = chromadb.PersistentClient(path="./ctrl_chroma_db")
collection = chroma_client.get_or_create_collection(name="study_materials")

app = FastAPI(title="Ctrl Backend - Final Hackathon Build")

# Global In-Memory State Tracker
class ActiveSessionData:
    def __init__(self):
        self.is_active: bool = False
        self.start_time: float = 0.0
        self.study_minutes: int = 0
        self.break_minutes: int = 0
        self.ai_adjusted_break: bool = False
        self.blocked_apps: List[str] = []
        self.current_material: str = ""

current_session = ActiveSessionData()


# ==========================================
# 2. DATA MODELS (Pydantic)
# ==========================================

class CreateSessionRequest(BaseModel):
    study_minutes: int
    break_minutes: Optional[int] = 0 
    ai_adjusted_break: bool = False   
    pages_to_study: int = 1  
    blocked_apps: List[str]
    current_material: str

class SessionActionRequest(BaseModel):
    action: str 

class QuizRequest(BaseModel):
    filename: str
    start_page: int
    end_page: int

class QuizVerificationRequest(BaseModel):
    total_questions: int
    correct_answers: int


# ==========================================
# 3. PHASE 1: ASYNCHRONOUS INGESTION
# ==========================================

@app.post("/upload-syllabus/")
async def upload_syllabus(file: UploadFile = File(...)):
    """Parses a PDF, chunks it by page, gets embeddings, and stores in ChromaDB."""
    reader = PdfReader(file.file)
    documents, metadatas, ids = [], [], []
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            documents.append(text)
            metadatas.append({"page": page_num + 1, "filename": file.filename})
            ids.append(f"{file.filename}_page_{page_num + 1}")

    if not documents:
        raise HTTPException(status_code=400, detail="Could not extract text from the PDF.")

    # Call Fireworks for Nomic Embeddings
    embedding_response = client.embeddings.create(
        model="nomic-ai/nomic-embed-text-v1.5",
        input=documents
    )
    embeddings = [data.embedding for data in embedding_response.data]

    # Store locally
    collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
    
    return {"status": "success", "message": f"Ingested {len(documents)} pages into ChromaDB."}


# ==========================================
# 4. PHASE 2: REAL-TIME GENERATION & GRADING
# ==========================================

@app.post("/generate-quiz/")
async def generate_quiz(request: QuizRequest):
    """Fetches text from ChromaDB and generates a JSON quiz via Fireworks LLM."""
    results = collection.get(
        where={
            "$and": [
                {"filename": request.filename},
                {"page": {"$gte": request.start_page}},
                {"page": {"$lte": request.end_page}}
            ]
        }
    )
    
    if not results['documents']:
        raise HTTPException(status_code=404, detail="No text found for those pages.")
        
    context_text = "\n".join(results['documents'])
    
    system_prompt = (
        "You are an educational AI. Based ONLY on the provided context, generate a 5-question multiple choice quiz. "
        "You MUST output your response in valid JSON format with the following structure: "
        "{'quiz': [{'question': '...', 'options': ['A', 'B', 'C', 'D'], 'answer': 'Exact string of correct option'}]} "
        "Do not include any conversational text."
    )
    
    chat_completion = client.chat.completions.create(
        model="accounts/fireworks/models/gemma-4-31b-it",
        response_format={"type": "json_object"}, 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context_text}"}
        ]
    )
    
    try:
        return json.loads(chat_completion.choices[0].message.content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="LLM failed to return valid JSON.")


@app.post("/session/verify-quiz")
async def verify_quiz_score(request: QuizVerificationRequest):
    """Verifies if the user scored >= 50% to unlock their intercepted apps."""
    if request.total_questions <= 0:
        raise HTTPException(status_code=400, detail="Total questions must be greater than 0.")
        
    score_percentage = (request.correct_answers / request.total_questions) * 100
    
    if score_percentage >= 50.0:
        current_session.is_active = False  # Shut down the OS lock
        return {
            "passed": True,
            "score_percentage": score_percentage,
            "message": f"Passed with {score_percentage:.1f}%. App block deactivated."
        }
    
    return {
        "passed": False,
        "score_percentage": score_percentage,
        "message": f"Failed with {score_percentage:.1f}%. You need at least 50% to unlock. Try again!"
    }


# ==========================================
# 5. PHASE 3: SESSION STATE MANAGEMENT
# ==========================================

@app.get("/session/status")
async def get_session_status():
    """Kotlin calls this on dashboard load to determine the UI state."""
    if not current_session.is_active:
        return {"has_ongoing_session": False}
    
    elapsed_seconds = time.time() - current_session.start_time
    elapsed_minutes = int(elapsed_seconds // 60)
    
    return {
        "has_ongoing_session": True,
        "elapsed_minutes": elapsed_minutes,
        "study_minutes": current_session.study_minutes,
        "break_minutes": current_session.break_minutes,
        "ai_adjusted_break": current_session.ai_adjusted_break,
        "blocked_apps": current_session.blocked_apps,
        "current_material": current_session.current_material
    }


@app.post("/session/start")
async def start_session(request: CreateSessionRequest):
    """Instantiates a session and calculates cognitive load if AI break is enabled."""
    if current_session.is_active:
        raise HTTPException(status_code=400, detail="A session is already running.")
    
    current_session.is_active = True
    current_session.start_time = time.time()
    current_session.study_minutes = request.study_minutes
    current_session.blocked_apps = request.blocked_apps
    current_session.current_material = request.current_material
    current_session.ai_adjusted_break = request.ai_adjusted_break
    
    study_density_flag = "standard"
    
    if request.ai_adjusted_break:
        # Calculate Study Density (pages per minute)
        study_density = request.pages_to_study / max(request.study_minutes, 1)
        
        # Determine Intensity Multiplier based on Cognitive Load
        if study_density >= 1.0:
            intensity_multiplier = 1.5   # Extreme Cramming
            study_density_flag = "cramming"
        elif study_density > 0.5:
            intensity_multiplier = 1.2   # Heavy Load
            study_density_flag = "heavy"
        elif study_density < 0.2:
            intensity_multiplier = 0.8   # Light Reading
            study_density_flag = "light"
        else:
            intensity_multiplier = 1.0   # Standard Pace
            
        # Apply to Base Ratio (20% of study time is standard break)
        base_break_minutes = request.study_minutes * 0.20
        calculated_break = int(base_break_minutes * intensity_multiplier)
        
        # Enforce Guardrails (Min 3 mins, Max 50% of total study time)
        max_allowed_break = int(request.study_minutes * 0.5)
        current_session.break_minutes = max(3, min(calculated_break, max_allowed_break))
        
    else:
        current_session.break_minutes = request.break_minutes
    
    return {
        "status": "success", 
        "message": "Ctrl intercept activated.",
        "allocated_break_minutes": current_session.break_minutes,
        "study_density_flag": study_density_flag
    }


@app.post("/session/verify-action")
async def verify_session_action(request: SessionActionRequest):
    """Checks the strict 5-minute grace period before allowing modification."""
    if not current_session.is_active:
        raise HTTPException(status_code=400, detail="No active session found.")
    
    elapsed_seconds = time.time() - current_session.start_time
    elapsed_minutes = elapsed_seconds / 60.0
    
    if elapsed_minutes < 5.0:
        if request.action == "cancel":
            current_session.is_active = False
        return {
            "quiz_required": False,
            "message": f"Action allowed without quiz. Elapsed time: {int(elapsed_seconds)}s"
        }
    
    return {
        "quiz_required": True,
        "message": "Grace period expired. A RAG quiz must be passed to authorize this modification."
    }
