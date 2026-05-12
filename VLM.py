import torch
from transformers import AutoModelForCausalLM
from PIL import Image

model_id = "vikhyatk/moondream2"
revision = "2025-06-21"

print("Moondream2 모델 로드 중...")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    revision=revision,
    trust_remote_code=True,
    torch_dtype=torch.float32
)

model = model.to("cpu")
model.eval()

image_path = "analysis/Original.png"
image = Image.open(image_path).convert("RGB")

question = (
    "Describe this chest x-ray image. "
    "Do not provide a clinical diagnosis. "
    "Only describe visible findings that may be relevant to pneumonia."
)

print("이미지 분석 중...")

result = model.query(
    image,
    question,
    settings={
        "variant": None,
        "temperature": 0.2,
        "max_tokens": 256,
        "top_p": 0.3
    }
)

print("\n" + "=" * 60)
print("VLM 출력 결과")
print("-" * 60)
print(result["answer"])
print("=" * 60)