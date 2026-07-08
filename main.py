import base64
import uuid
import json
import chromadb
import fitz  # PyMuPDF for handling PDFs
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from openai import OpenAI

app = FastAPI(title="Gemma 4 Omni-Channel RAG Engine")

# Initialize OpenAI client pointed to Fireworks AI
FIREWORKS_API_KEY = "YOUR_FIREWORKS_API_KEY"
client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1", 
    api_key=FIREWORKS_API_KEY
)

# ==========================================
# 🛠️ TESTING TOGGLE (Change to False for final build)
# ==========================================
DEBUG_MODE = True  

# Dedicated On-Demand Path (For Final Submission)
GEMMA_DEPLOYMENT = "accounts/your_username/deployments/your_deployment_name"

# The Absolute Best Serverless Fallbacks available right now
SERVERLESS_VISION = "accounts/fireworks/models/qwen3p7-plus"
SERVERLESS_TEXT = "accounts/fireworks/models/deepseek-v4-flash"
# ==========================================

# Initialize ChromaDB persistent storage locally on the server
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="textbook_materials")

def encode_image(file_bytes: bytes) -> str:
    """Converts raw image bytes to a base64 string for the vision model."""
    return base64.b64encode(file_bytes).decode('utf-8')

@app.post("/upload-material/")
async def upload_material(title: str = Form(...), file: UploadFile = File(...)):
    """
    Phase 1: Receives an image or PDF.
    - If Image: Extracts text using the Vision LLM.
    - If PDF: Natively parses the digital text.
    Chunks the text and saves it alongside metadata into ChromaDB.
    """
    try:
        file_bytes = await file.read()
        content_type = file.content_type
        extracted_text = ""
        
        # --- 1. ROUTE BASED ON FILE TYPE ---
        if "image" in content_type:
            # Handle Physical Textbook Photos via Vision LLM
            base64_image = encode_image(file_bytes)
            active_vision_model = SERVERLESS_VISION if DEBUG_MODE else GEMMA_DEPLOYMENT
            
            vision_completion = client.chat.completions.create(
                model=active_vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract all academic text from this textbook page perfectly. Output only the plain text. Do not include markdown formatting."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ]
            )
            extracted_text = vision_completion.choices[0].message.content
            
        elif "pdf" in content_type:
            # Handle Digital PDFs via PyMuPDF (Lightning Fast)
            pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
            pdf_text_pages = []
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                # Extract text and append to our page array
                pdf_text_pages.append(page.get_text("text"))
                
            extracted_text = "\n\n".join(pdf_text_pages)
            pdf_document.close()
            
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload an image or a PDF.")

        # --- 2. VALIDATE EXTRACTION ---
        if not extracted_text or not extracted_text.strip():
            raise HTTPException(status_code=500, detail="Failed to extract text from the file.")

        # --- 3. TEXT CHUNKING ---
        # Split by double line breaks to isolate paragraphs/sections
        chunks = [chunk.strip() for chunk in extracted_text.split("\n\n") if len(chunk.strip()) > 20]
        
        if not chunks:
            return {"status": "success", "title": title, "chunks_saved": 0, "message": "No meaningful text detected."}

        # --- 4. PREPARE & SAVE TO CHROMADB ---
        chunk_ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"title": title} for _ in chunks]
        
        collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=chunk_ids
        )
        
        return {
            "status": "success", 
            "title": title, 
            "chunks_saved": len(chunks),
            "file_type_processed": "PDF" if "pdf" in content_type else "Image",
            "mode": "DEBUG_SERVERLESS" if DEBUG_MODE else "PRODUCTION_GEMMA"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/materials/")
async def list_materials():
    """
    Phase 2: The Intercept.
    Returns a list of all unique material titles saved in the database
    so the Kotlin frontend can render the selection checkboxes.
    """
    try:
        # Get all metadata from ChromaDB
        db_data = collection.get(include=["metadatas"])
        metadatas = db_data.get("metadatas", [])
        
        # Extract unique titles using a set
        unique_titles = list(set([meta["title"] for meta in metadatas if meta and "title" in meta]))
        
        return {"saved_materials": unique_titles}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-quiz/")
async def generate_quiz(topic: str = Form(...), selected_titles: str = Form(...)):
    """
    Phase 3: Queries ChromaDB by metadata titles and semantic topic,
    then feeds grounded text context to the LLM to output a structured quiz.
    Note: selected_titles should be passed as a comma-separated string from Android.
    """
    try:
        title_list = [t.strip() for t in selected_titles.split(",") if t.strip()]
        
        if not title_list:
            raise HTTPException(status_code=400, detail="You must select at least one material to generate a quiz.")

        # 1. Query ChromaDB using metadata filters ($in allows multiple titles) and semantic similarity
        results = collection.query(
            query_texts=[topic],
            n_results=5, # Pull top 5 most relevant paragraphs
            where={"title": {"$in": title_list}} 
        )
        
        # Flatten retrieved text documents into a single context string
        retrieved_documents = results.get("documents", [[]])[0]
        context_text = "\n---\n".join(retrieved_documents)
        
        if not context_text:
            return JSONResponse(
                status_code=400,
                content={"error": "INSUFFICIENT_DATA", "message": "No relevant material found for this topic in the selected chapters."}
            )

        # 2. Establish strict grounding rules using the instruction-tuned system prompt
        system_prompt = (
            "You are an academic instructor. Your ONLY task is to generate multiple-choice questions based EXCLUSIVELY on the provided Context.\n"
            "Rules:\n"
            "1. Use ONLY the information provided in the Context to form your questions and answers.\n"
            "2. If the Context does not contain enough information to reliably answer a question, return the string 'INSUFFICIENT_DATA' instead of fabricating data.\n"
            "3. Do not use outside knowledge or training data.\n"
            "4. Return the output as a raw JSON array matching this structure exactly:\n"
            "[\n"
            "  {\n"
            "    \"question\": \"...\",\n"
            "    \"options\": [\"A\", \"B\", \"C\", \"D\"],\n"
            "    \"correct_answer\": \"...\"\n"
            "  }\n"
            "]"
        )
        
        user_content = f"Context:\n{context_text}\n\nTask: Generate a 3-question multiple-choice quiz based on the topic: '{topic}'."

        # 3. Execute reasoning step with Text LLM
        active_text_model = SERVERLESS_TEXT if DEBUG_MODE else GEMMA_DEPLOYMENT

        quiz_completion = client.chat.completions.create(
            model=active_text_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.2  # Low temperature forces strict adherence to rules and schemas
        )
        
        raw_output = quiz_completion.choices[0].message.content.strip()
        
        # Guardrail: Check if the model triggered the out-of-bounds safety keyword
        if "INSUFFICIENT_DATA" in raw_output:
            return JSONResponse(
                status_code=400, 
                content={"error": "INSUFFICIENT_DATA", "message": "The material does not contain enough data on this topic."}
            )

        # 4. Clean formatting wrappers if the model accidentally includes markdown json blocks
        if raw_output.startswith("```json"):
            raw_output = raw_output.replace("```json", "", 1).rstrip("```").strip()
        elif raw_output.startswith("```"):
            raw_output = raw_output.replace("```", "", 1).rstrip("```").strip()

        # Parse string safely into native JSON array for the mobile client
        quiz_json = json.loads(raw_output)
        return quiz_json

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Model failed to output a valid JSON format. Try again.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
