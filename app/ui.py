# app/ui.py

import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Vietnamese Math Solver", layout="centered")

API_URL = "http://localhost:8000/solve"  # or public IP if deployed

st.title("📘 Vietnamese High School Math Solver")

st.markdown(
    """
    Upload a math problem image (with equations/geometry) and describe your question in Vietnamese.  
    This app will extract the problem, search for similar examples, and use an LLM to answer it step-by-step.
    """
)

uploaded_file = st.file_uploader("📷 Upload math image (PNG/JPG)", type=["png", "jpg", "jpeg"])

question = st.text_area("✍️ Enter your math question (in Vietnamese)", height=100)

submit_button = st.button("🧠 Solve Problem")

if submit_button:
    if not uploaded_file or not question.strip():
        st.warning("Vui lòng tải ảnh và nhập câu hỏi.")
    else:
        with st.spinner("Đang xử lý..."):
            files = {"image": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            data = {"question": question}

            try:
                response = requests.post(API_URL, files=files, data=data)
                response.raise_for_status()
                result = response.json()

                st.success("✅ Đã giải xong!")

                with st.expander("📄 Câu hỏi gốc"):
                    st.write(result["question"])

                with st.expander("🔍 Văn bản trích xuất từ ảnh (OCR)"):
                    st.write(result["ocr_text"])

                with st.expander("📚 Ví dụ tương tự được truy xuất"):
                    if result["retrieved_examples"]:
                        for i, ex in enumerate(result["retrieved_examples"], 1):
                            st.markdown(f"**Ví dụ {i}:** {ex}")
                    else:
                        st.write("Không tìm thấy ví dụ nào.")

                st.markdown("## 💡 Đáp án:")
                st.markdown(f"```text\n{result['answer']}\n```")

            except requests.exceptions.RequestException as e:
                st.error(f"Lỗi API: {e}")
