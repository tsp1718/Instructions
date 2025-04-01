from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pdfminer.high_level import extract_text
import docx2txt
from PIL import Image
import base64
import re
import io
import os
from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_txt(file):
    return file.file.read().decode("utf-8")

def extract_pdf(file):
    text = extract_text(file.file)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_docx(file):
    text = docx2txt.process(file.file)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_img(image_file):
    image_bytes = image_file.read()
    img = Image.open(io.BytesIO(image_bytes))
    content_type = f"image/{img.format.lower()}"
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{content_type};base64,{base64_image}"

def generate_prompt(developer_type, usecase_document, usecase_document_metadata,
                     meeting_notes, meeting_notes_metadata, screenshot_metadata, developer_instructions):
    prompt = f"""**Role:** You are an expert {developer_type} developer. You have been assigned a task with relevant context.

**Objective:** Carefully analyze the provided context (Use Case Document, Meeting Notes, Screenshot, and their Metadata) and execute the 'Instructions for Developer' to generate the required output. Synthesize information from all sections as needed.

**BEGIN TASK CONTEXT**

### 1. Use Case Document & Metadata
**Document:**
{usecase_document}
**Metadata:**
{usecase_document_metadata}

### 2. Meeting Notes & Metadata
**Meeting Notes:**
{meeting_notes}
**Metadata:**
{meeting_notes_metadata}

### 3. Screenshot/Image & Metadata
**Metadata:**
{screenshot_metadata}

**END TASK CONTEXT**

### Instructions for Developer (Your Task)
Based *strictly* on the context provided above, complete the following task:

{developer_instructions}

### Output Requirements
- Generate *only* the output requested in the 'Instructions for Developer'.
- Ensure your output directly addresses the instructions and accurately incorporates details from *all* context sections.
- Do not add conversational explanations, introductions, or summaries unless explicitly requested.
- Focus solely on fulfilling the task.
"""
    return prompt.strip()

def process_image_and_generate_response(prompt, image_file):
    google_api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.1,
        max_tokens=10000,
        google_api_key=google_api_key
    )
    data_uri = extract_img(image_file.file)
    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": data_uri}}
    ])
    response = llm.invoke([message])
    return response.content

@app.post("/instruct")
async def process_file(
    developer_type: str = Form(...),
    usecase_document: UploadFile = File(...),
    usecase_document_metadata: str = Form(...),
    meeting_notes: UploadFile = File(...),
    meeting_notes_metadata: str = Form(...),
    screenshot_metadata: str = Form(...),
    developer_instructions: str = Form(...),
    image_file: UploadFile = File(...)
):
    try:
        if usecase_document.filename.endswith('.pdf'):
            usecase_doc_text = extract_pdf(usecase_document)
        elif usecase_document.filename.endswith('.docx'):
            usecase_doc_text = extract_docx(usecase_document)
        else:
            usecase_doc_text = extract_txt(usecase_document)
        
        if meeting_notes.filename.endswith('.pdf'):
            meeting_notes_text = extract_pdf(meeting_notes)
        elif meeting_notes.filename.endswith('.docx'):
            meeting_notes_text = extract_docx(meeting_notes)
        else:
            meeting_notes_text = extract_txt(meeting_notes)
        
        prompt = generate_prompt(developer_type, usecase_doc_text, usecase_document_metadata,
                                 meeting_notes_text, meeting_notes_metadata, screenshot_metadata, developer_instructions)
        response = process_image_and_generate_response(prompt, image_file)
        return {"response": response}



    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
