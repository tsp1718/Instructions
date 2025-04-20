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
from typing import Optional # Import Optional
import uvicorn
# --- Load Environment Variables ---
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    print("Warning: GOOGLE_API_KEY environment variable not set.")
    # Consider raising an error or exiting if the key is essential
    # raise ValueError("GOOGLE_API_KEY environment variable not set.")


# --- FastAPI App Initialization ---
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions ---

def extract_txt(file: UploadFile):
    """Extracts text from a plain text file."""
    try:
        content = file.file.read().decode("utf-8")
        return content
    except Exception as e:
        print(f"Error reading text file {file.filename}: {e}")
        return f"Error reading file: {e}" # Return error message instead of raising

def extract_pdf(file: UploadFile):
    """Extracts text from a PDF file."""
    try:
        # pdfminer requires a file-like object supporting read() and seek()
        # Use io.BytesIO to wrap the file stream
        pdf_stream = io.BytesIO(file.file.read())
        text = extract_text(pdf_stream)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        print(f"Error reading PDF file {file.filename}: {e}")
        return f"Error reading file: {e}" # Return error message

def extract_docx(file: UploadFile):
    """Extracts text from a DOCX file."""
    try:
        # docx2txt can sometimes work directly with the SpooledTemporaryFile
        # but wrapping in BytesIO is safer if issues arise
        docx_stream = io.BytesIO(file.file.read())
        text = docx2txt.process(docx_stream)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        print(f"Error reading DOCX file {file.filename}: {e}")
        return f"Error reading file: {e}" # Return error message

def extract_img_data_uri(image_file: UploadFile):
    """Encodes image file content to a base64 data URI."""
    try:
        image_bytes = image_file.file.read()
        # Use Pillow to determine the image format safely
        with Image.open(io.BytesIO(image_bytes)) as img:
             img_format = img.format or 'jpeg' # Default to jpeg if format is unknown
        content_type = f"image/{img_format.lower()}"
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{content_type};base64,{base64_image}"
    except Exception as e:
        print(f"Error processing image file {image_file.filename}: {e}")
        return None # Indicate failure

def generate_prompt(developer_type, usecase_document, usecase_document_metadata,
                    meeting_notes, meeting_notes_metadata, screenshot_metadata, developer_instructions):
    """Generates the prompt for the LLM, handling potentially missing context."""

    # Use placeholders for missing information
    usecase_doc_text = usecase_document if usecase_document else "N/A"
    usecase_meta_text = usecase_document_metadata if usecase_document_metadata else "N/A"
    meeting_notes_text = meeting_notes if meeting_notes else "N/A"
    meeting_meta_text = meeting_notes_metadata if meeting_notes_metadata else "N/A"
    screenshot_meta_text = screenshot_metadata if screenshot_metadata else "N/A"
    dev_type = developer_type if developer_type else "Software" # Default developer type

    prompt = f"""**Role:** You are an expert {dev_type} developer. You have been assigned a task with relevant context.

**Objective:** Carefully analyze the provided context (Use Case Document, Meeting Notes, Screenshot Metadata, if available) and execute the 'Instructions for Developer' to generate the required output. Synthesize information from all provided sections as needed. If a section is marked 'N/A', it was not provided.

**BEGIN TASK CONTEXT**

### 1. Use Case Document & Metadata
**Document:**
{usecase_doc_text}
**Metadata:**
{usecase_meta_text}

### 2. Meeting Notes & Metadata
**Meeting Notes:**
{meeting_notes_text}
**Metadata:**
{meeting_meta_text}

### 3. Screenshot/Image & Metadata (Image provided separately if available)
**Metadata:**
{screenshot_meta_text}

**END TASK CONTEXT**

### Instructions for Developer (Your Task)
Based *strictly* on the context provided above (and the image, if supplied), complete the following task:

{developer_instructions}

### Output Requirements
- Generate *only* the output requested in the 'Instructions for Developer'.
- Ensure your output directly addresses the instructions and accurately incorporates details from the provided context sections.
- Do not add conversational explanations, introductions, or summaries unless explicitly requested.
- Focus solely on fulfilling the task.
"""
    return prompt.strip()

def generate_llm_response(prompt: str, image_data_uri: Optional[str] = None):
    """Generates response from the LLM, handling optional image input."""
    if not google_api_key:
        raise HTTPException(status_code=500, detail="Server configuration error: Google API Key not set.")

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", # Or your preferred model
            temperature=0.1,
            max_tokens=10000, # Adjust as needed
            google_api_key=google_api_key
        )

        content = [{"type": "text", "text": prompt}]
        if image_data_uri:
            content.append({"type": "image_url", "image_url": {"url": image_data_uri}})

        message = HumanMessage(content=content)
        response = llm.invoke([message])
        return response.content

    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        # Re-raise as HTTPException to send proper error response to client
        raise HTTPException(status_code=500, detail=f"Error generating response from AI model: {e}")


# --- FastAPI Endpoint ---

@app.post("/instruct")
async def process_instruction(
    # --- Mandatory Parameter ---
    developer_instructions: str = Form(...),

    # --- Optional Parameters ---
    developer_type: Optional[str] = Form(None),
    usecase_document: Optional[UploadFile] = File(None),
    usecase_document_metadata: Optional[str] = Form(None),
    meeting_notes: Optional[UploadFile] = File(None),
    meeting_notes_metadata: Optional[str] = Form(None),
    screenshot_metadata: Optional[str] = Form(None),
    image_file: Optional[UploadFile] = File(None) # Changed from image_file to screenshot
):
    """
    Processes developer instructions with optional context documents and an image.
    `developer_instructions` is the only mandatory field.
    """
    usecase_doc_text = None
    meeting_notes_text = None
    image_data_uri = None

    try:
        # --- Process Optional Use Case Document ---
        if usecase_document and usecase_document.filename:
            print(f"Processing use case document: {usecase_document.filename}")
            if usecase_document.filename.endswith('.pdf'):
                usecase_doc_text = extract_pdf(usecase_document)
            elif usecase_document.filename.endswith('.docx'):
                usecase_doc_text = extract_docx(usecase_document)
            elif usecase_document.filename.endswith('.txt'): # Added explicit check for txt
                usecase_doc_text = extract_txt(usecase_document)
            else:
                # Handle unsupported file types gracefully
                print(f"Unsupported file type for use case document: {usecase_document.filename}")
                usecase_doc_text = f"Unsupported file type: {usecase_document.content_type}"
            # Ensure file pointer is reset if needed (though BytesIO usually handles this)
            await usecase_document.seek(0)


        # --- Process Optional Meeting Notes ---
        if meeting_notes and meeting_notes.filename:
            print(f"Processing meeting notes: {meeting_notes.filename}")
            if meeting_notes.filename.endswith('.pdf'):
                meeting_notes_text = extract_pdf(meeting_notes)
            elif meeting_notes.filename.endswith('.docx'):
                meeting_notes_text = extract_docx(meeting_notes)
            elif meeting_notes.filename.endswith('.txt'): # Added explicit check for txt
                 meeting_notes_text = extract_txt(meeting_notes)
            else:
                 # Handle unsupported file types gracefully
                print(f"Unsupported file type for meeting notes: {meeting_notes.filename}")
                meeting_notes_text = f"Unsupported file type: {meeting_notes.content_type}"
            # Ensure file pointer is reset
            await meeting_notes.seek(0)

        # --- Process Optional Image File ---
        if image_file and image_file.filename:
            print(f"Processing image file: {image_file.filename}")
            image_data_uri = extract_img_data_uri(image_file)
            if image_data_uri is None:
                # Handle image processing error if needed, maybe add to metadata?
                screenshot_metadata = (screenshot_metadata + " (Error processing image)" if screenshot_metadata else "Error processing image")
            # Ensure file pointer is reset
            await image_file.seek(0)


        # --- Generate Prompt ---
        # Pass potentially None or extracted text values
        prompt = generate_prompt(
            developer_type,
            usecase_doc_text, # Could be None or extracted text
            usecase_document_metadata, # Could be None or provided string
            meeting_notes_text, # Could be None or extracted text
            meeting_notes_metadata, # Could be None or provided string
            screenshot_metadata, # Could be None or provided string
            developer_instructions # Mandatory
        )

        # --- Generate LLM Response (Conditionally with Image) ---
        response_content = generate_llm_response(prompt, image_data_uri)

        return {"response": response_content}

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc() # Log the full traceback for debugging
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

# --- Run the App (for local development) ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
