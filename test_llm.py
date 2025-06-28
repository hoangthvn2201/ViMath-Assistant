from app.llm import LLMEngine
from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv('HUGGINGFACE_TOKEN')
login(token=hf_token)

llm = LLMEngine(
    backend="gemini",  # or "phi-2"
    model_name_or_path="phi-2",  # ignored for gemini
    gemini_api_key=os.getenv("GEMINI_API_KEY")
)

retrieved = [
    "Tìm nghiệm của phương trình x^2 - 3x + 2 = 0",
    "Giải phương trình x^2 - 4x + 3 = 0"
]

prompt = llm.build_prompt(
    user_question="Tìm nghiệm của phương trình x^2 - 5x + 6 = 0",
    retrieved_examples=retrieved,
    category="algebra"
)

print(prompt)
answer = llm.generate_answer(prompt)
print("Generated Answer:\n", answer)


