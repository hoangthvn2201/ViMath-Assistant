# app/prompts/cot_templates.py

from typing import List

COT_TEMPLATES = {
    "algebra": [
        {
            "question": "Tìm nghiệm của phương trình x^2 - 5x + 6 = 0",
            "reasoning": "Phương trình là bậc hai, ta phân tích thành (x - 2)(x - 3) = 0. Suy ra nghiệm là x = 2 hoặc x = 3."
        },
        {
            "question": "Giải phương trình x^2 - 4 = 0",
            "reasoning": "Ta chuyển vế: x^2 = 4. Lấy căn hai hai vế, ta được x = ±2."
        }
    ],
    "geometry": [
        {
            "question": "Tính diện tích hình tròn có bán kính 5 cm",
            "reasoning": "Diện tích hình tròn là πr^2. Với r = 5, ta có S = π * 5^2 = 25π."
        }
    ],
    "word_problem": [
        {
            "question": "Một xe máy đi từ A đến B với vận tốc 40 km/h. Hỏi sau 2 giờ xe đi được bao nhiêu km?",
            "reasoning": "Quãng đường = Vận tốc × Thời gian = 40 × 2 = 80 km."
        }
    ]
}

def get_few_shot_examples(category: str = "algebra", n: int = 2) -> str:
    """
    Returns few-shot formatted prompt from template examples.

    Args:
        category (str): One of the categories in COT_TEMPLATES
        n (int): Number of examples to return

    Returns:
        str: Formatted CoT prompt segment
    """
    examples = COT_TEMPLATES.get(category, [])[:n]
    formatted = ""
    for i, ex in enumerate(examples, 1):
        formatted += f"Example {i}: \n"
        formatted += f"Question: {ex['question']}\n"
        formatted += f"Answer: {ex['reasoning']}\n\n"
    return formatted.strip()



def generate_prompt_cot(user_question: str, retrieved_examples: List[str], category: str = "") -> str:
    """
    Generate a CoT (Chain-of-Thought) prompt from the user's question and retrieved examples.

    Args:
        user_question (str): The math problem provided by the user.
        retrieved_examples (List[str]): Similar math problems and solutions retrieved from the index.
        category (str): (Optional) Category such as "Algebra", "Geometry", etc.

    Returns:
        str: Prompt formatted for LLM with examples and reasoning instructions.
    """

    intro = (
        f"You are a Vietnamese high school math assistant. "
        f"Use step-by-step logical reasoning (chain-of-thought) to solve problems. "
        f"{'This problem is in the category: ' + category if category else ''}\n"
        "Below are similar problems and their solutions:\n"
    )

    few_show_examples = get_few_shot_examples(category=category, n=2)

    examples = ""
    for i, example in enumerate(retrieved_examples, 1):
        examples += f"Example {i}:\n{example.strip()}\n\n"

    final_prompt = (
        f"{intro}\n"
        "Example based on your problem category:\n"
        f"{few_show_examples}\n"
        "Example retrieved from database:\n"
        f"{examples}"
        "Now solve this problem:\n"
        f"{user_question.strip()}\n"
        "Let's think step by step:"
    )

    return final_prompt