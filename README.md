# 📘 Vietnamese High School Math Assistant (ViMath)

A **multimodal retriever-based system** to assist Vietnamese high school students in solving math problems using image and text inputs. Powered by lightweight language and vision models, this system is designed for **local deployment**, **low-resource environments**, and **scalable integration** with future models or tools.

---

## 🎯 Project Purpose

The ViMath Assistant aims to:

- Support Vietnamese high school students in solving math problems.
- Leverage OCR, text/image embeddings, and retrieval-augmented generation.
- Provide accurate, step-by-step explanations with strong logic reasoning.
- Run on resource-constrained devices such as laptops with low-end GPUs or CPU-only setups.

---

## 🏗️ System Architecture

```text
User Input (Image + Optional Text)
            │
            ▼
     [OCR Module]  
 (Vietnamese OCR via PaddleOCR)
            │
            ▼
 [Text + Image Embedding Encoders]  
 (Sentence-BERT + Fine-tuned CLIP)
            │
            ▼
    [Vector Retriever (FAISS)]  
 └─ Retrieves Top-K Relevant Examples
            │
            ▼
 [Prompt Composer + CoT Examples]
            │
            ▼
  [Lightweight LLM Generator]  
 (e.g., Phi-2, TinyLLaMA, PhoGPT)
            │
            ▼
Answer with Reasoning Explanation
```

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/vimath-assistant.git
cd vimath-assistant 
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Start Backend Service
```bash
uvicorn app.main:app --reload
```
### 4. Start Frontend UI
```bash
streamlit run app/ui.py
```

