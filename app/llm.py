# app/llm.py

from huggingface_hub import login
from dotenv import load_dotenv
import os
import requests
from typing import List, Literal, Optional
from prompts.cot_templates import generate_prompt_cot

load_dotenv()


LLM_BACKENDS = Literal["phi-2", "gemini"]

class LLMEngine:
    def __init__(
        self,
        model_name_or_path: str = "phi-2",
        backend: LLM_BACKENDS = "phi-2",
        max_tokens: int = 512,
        temperature: float = 0.3,
        gemini_api_key: Optional[str] = None
    ):
        self.backend = backend
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model_name_or_path = model_name_or_path
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")

        if self.backend == "phi-2":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            from huggingface_hub import login

            hf_token = os.getenv('HUGGINGFACE_TOKEN')
            login(token=hf_token)

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, device_map="auto", torch_dtype=torch.float16
            )
            self.model.eval()

        elif self.backend == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.prompts import PromptTemplate

            if not self.gemini_api_key:
                raise ValueError("GEMINI_API_KEY must be set for Gemini backend.")
            else:
                self.model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=self.gemini_api_key, temperature = 0.5)

    def build_prompt(self, user_question: str, retrieved_examples: List[str] = None, category: str = "") -> str:
        """
        Compose the prompt using CoT examples and retrieved samples.

        Args:
            user_question (str): The question from OCR or user input.
            retrieved_examples (List[str]): Similar problems (retrieved).
            category (str): Type of problem (for CoT template selection)

        Returns:
            str: Final composed prompt
        """
        return generate_prompt_cot(user_question, retrieved_examples, category)

    def generate_answer(self, prompt: str) -> str:
        if self.backend == "phi-2":
            return self._generate_phi(prompt)
        elif self.backend == "gemini":
            return self._generate_gemini(prompt)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _generate_phi(self, prompt: str) -> str:
        import torch

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        output = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=True,
        )
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return decoded[len(prompt):].strip()

    def _generate_gemini(self, prompt: str) -> str:
        response = self.model.invoke(prompt)

        # data = response.json()
        # try:
        #     return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        # except (KeyError, IndexError):
        #     raise RuntimeError("Unexpected Gemini response format.")

        return response.content
        
