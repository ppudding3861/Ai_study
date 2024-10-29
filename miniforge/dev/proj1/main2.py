from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# 모델과 토크나이저 로드
model_name = "timpal0l/mdeberta-v3-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# 예시 컨텍스트와 질문
context = "인공지능은 컴퓨터 과학의 한 분야로, 기계가 인간처럼 학습, 추론, 문제 해결을 수행하도록 하는 기술입니다."
question = "오메가3는 무엇인가요?"

# 입력 데이터 토큰화
inputs = tokenizer(question, context, return_tensors="pt")

# 답변 예측
with torch.no_grad():
    outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

print("답변:", answer)
