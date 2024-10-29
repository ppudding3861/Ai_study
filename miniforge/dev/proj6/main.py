from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# 환경 변수에서 CORS 허용 URL 가져오기
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

# 모델과 토크나이저를 로드하고 GPU/CPU 설정
model_name = "illuni/illuni-llama-2-ko-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# GPU/CPU 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# FastAPI 객체 생성
app = FastAPI()

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # 환경 변수에서 가져온 URL을 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 엔드포인트 정의
@app.post("/chat/")
async def inference(text: str = Form(...)):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    output = model.generate(inputs["input_ids"], max_length=50, do_sample=True, top_p=0.9, temperature=0.7)
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"result": result}
