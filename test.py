from PIL import Image
from app.ocr import run_ocr_from_pil

img = Image.open(r"C:\Users\huyho\OneDrive\Desktop\MathRAG\image.png")
text, lines = run_ocr_from_pil(img)

print("Extracted Text:", text)
print("Lines:", lines)
