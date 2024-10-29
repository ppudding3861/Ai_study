from transformers import XGLMTokenizer, AutoModelForCausalLM

# 모델 및 토크나이저 로드
model_name = "facebook/xglm-564M"
tokenizer = XGLMTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name)

def answer_question(question):
    # 프롬프트를 질문-답변 형식으로 설정
    prompt = f"질문: {question}\n답변:"
    
    # 질문-답변 형식으로 입력 텍스트를 토큰화
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 답변 생성
    output = model.generate(
        **inputs, 
        max_length=50, 
        num_return_sequences=1, 
        temperature=0.7,    # 답변의 다양성 조절
        top_p=0.9,          # 상위 확률 조절
        do_sample=True      # 샘플링 기반 생성을 활성화
    )
    
    # 결과 디코딩 및 반환
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # 생성된 답변에서 질문 부분을 제외한 답변 텍스트만 반환
    answer = answer.split("답변:")[-1].strip()
    return answer

# 사용자가 질문을 입력하면 답변 출력
user_question = input("질문을 입력하세요: ")
answer = answer_question(user_question)
print("답변:", answer)
