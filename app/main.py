# main.py

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os

# from ocr import OCRProcessor
from ocr import run_ocr
from embeddings.text_encoder import TextEncoder
from embeddings.image_encoder import ImageEncoder
from retriever import Retriever
from llm import LLMEngine
from utils import clean_text, create_temp_image, safe_remove

# Initialize core components
# ocr_processor = OCRProcessor()
text_encoder = TextEncoder()
image_encoder = ImageEncoder()
# retriever = Retriever(text_encoder=text_encoder, image_encoder=image_encoder)
index_path = r"C:\Users\huyho\OneDrive\Desktop\MathRAG\data\faiss_index\math.index"
corpus_path = r"C:\Users\huyho\OneDrive\Desktop\MathRAG\data\processed\corpus.pkl"
retriever = Retriever(index_path=index_path, corpus_path=corpus_path, text_encoder=text_encoder)
# llm_engine = LLMEngine(model_name_or_path="phi-2", max_tokens=512)
# Use Gemini instead of Phi-2
llm_engine = LLMEngine(
    backend="gemini",  # or "phi-2"
    model_name_or_path="phi-2",  # ignored for gemini
    gemini_api_key=os.getenv("GEMINI_API_KEY")
)

# Initialize API
app = FastAPI(
    title="Vietnamese High School Math Solver",
    description="An assistant that answers Vietnamese math questions from image and text input using multimodal retrieval and LLM reasoning.",
    version="1.0.0"
)

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/solve")
async def solve_math_problem(
    image: UploadFile = File(...),
    question: str = Form(...)
):
    try:
        # Load image
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # OCR (optional – can be used later to improve embedding context)
        # ocr_text = ocr_processor.run_ocr(pil_image)
        ocr_text = run_ocr(pil_image)
        cleaned_ocr = clean_text(ocr_text)

        # Encode + retrieve related examples
        retrieved = retriever.retrieve(
            question=question,
            image=pil_image,
            top_k=3
        )

        # Build prompt + generate answer
        prompt = llm_engine.build_prompt(
            user_question=question,
            retrieved_examples=retrieved,
            category="algebra"  # Optional: can infer from question type
        )

        answer = llm_engine.generate_answer(prompt)

        return {
            "question": question,
            "ocr_text": cleaned_ocr,
            "retrieved_examples": retrieved,
            "answer": answer
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )