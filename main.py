import base64
import uuid
import json
import chromadb
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from openai import OpenAI

app = FastAPI(title="Gemma 4 RAG Quiz Engine")

# Initialize OpenAI client pointed to Fireworks AI
# Replace with your actual Fireworks API key
FIREWORKS_API_KEY = "YOUR_FIREWORKS_API_KEY"
client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1", 
    api_key=FIREWORKS_API_KEY
)

# Use your exact chosen Gemma 4 deployment path
GEMMA_DEPLOYMENT = "accounts/your_username/deployments/your_deployment_name"

# Initialize ChromaDB persistent storage locally on the server
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="textbook_materials")

def encode_image(file_bytes: bytes) -> str:
    """Converts raw image bytes to a base64 string for the vision model."""
    return base64.b64encode(file_bytes).decode('utf-8')

@app.post("/upload-material/")
async def upload_material(title: str = Form(...), file: UploadFile = File(...)):
    """
    Phase 1: Receives a textbook image, extracts text using Gemma Vision,
    chunks it, and saves the text alongside metadata into ChromaDB.
    """
    try:
        # 1. Read and encode the incoming image file
        image_bytes = await file.read()
        base64_image = encode_image(image_bytes)
        
        # 2. Extract text using Gemma 4 Vision capability
        vision_completion = client.chat.completions.create(
            model=GEMMA_DEPLOYMENT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all academic text from this textbook page perfectly. Output only the plain text."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
        )
        extracted_text = vision_completion.choices[0].message.content
        
        if not extracted_text:
            raise HTTPException(status_code=500, detail="Failed to extract text from image.")

        # 3. Text Chunking (Simple paragraph splitter for hackathon velocity)
        chunks = [chunk.strip() for chunk in extracted_text.split("\n\n") if chunk.strip()]
        
        if not chunks:
            return {"status": "success", "title": title, "chunks_saved": 0, "message": "No text detected."}

        # 4. Prepare batch arrays for ChromaDB
        chunk_ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"title": title} for _ in chunks]
        
        # 5. Insert translated text chunks into the vector database
        collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=chunk_ids
        )
        
        return {
            "status": "success", 
            "title": title, 
            "chunks_saved": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-quiz/")
async def generate_quiz(topic: str = Form(...), title: str = Form(...)):
    """
    Phase 2: Queries ChromaDB by metadata title and semantic topic,
    then feeds grounded text context to Gemma 4 to output a structured quiz.
    """
    try:
        # 1. Query ChromaDB using metadata filters and semantic similarity query
        results = collection.query(
            query_texts=[topic],
            n_results=4,
            where={"title": title}  # Ensures search stays strictly inside this specific material
        )
        
        # Flatten retrieved text documents into a single context string
        retrieved_documents = results.get("documents", [[]])[0]
        context_text = "\n---\n".join(retrieved_documents)
        
        if not context_text:
            return JSONResponse(
                status_code=400,
                content={"error": "INSUFFICIENT_DATA", "message": "No relevant material found for this topic."}
            )

        # 2. Establish strict grounding rules using the instruction-tuned system prompt
        system_prompt = (
            "You are an academic instructor. Your ONLY task is to generate multiple-choice questions based EXCLUSIVELY on the provided Context.\n"
            "Rules:\n"
            "1. Use ONLY the information provided in the Context to form your questions and answers.\n"
            "2. If the Context does not contain enough information to reliably answer a question, return the string 'INSUFFICIENT_DATA' instead of fabricating data.\n"
            "3. Do not use outside knowledge or training data.\n"
            "4. Return the output as a raw JSON array matching this structure exactly: "
            "[{\"question\": \"...\", \"options\": [\"A\", \"B\", \"C\", \"D\"], \"correct_answer\": \"...\"}]"
        )
        
        user_content = f"Context:\n{context_text}\n\nTask: Generate a 3-question multiple-choice quiz based on the topic: '{topic}'."

        # 3. Execute reasoning step with Gemma 4
        quiz_completion = client.chat.completions.create(
            model=GEMMA_DEPLOYMENT,
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
