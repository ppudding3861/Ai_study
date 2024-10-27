
# step 1 : import module
from transformers import pipeline


# step 2 : create inference object(instance)
classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_qa_model")


from fastapi import FastAPI, Form

app = FastAPI()


@app.post("/inference/")
async def inference(text: str = Form()):
    # # step 3 : prepare data
    # text = "Tiffany Doll - Wine Makes Me Anal (31.03.2018)_1080p.mp4"


    # step 4 : inferece
    result = classifier(text)

    # step 5 : post processing
    print(result)

    return {"result": result}







'''
text

fast api 튜토리얼 폼데이터로 들어간다

복사 붙여넣기
패스워드 지우고
유저네임을 result로 변경

cls 에서 step1~2 복사 붙여넣기
step3~5

step3 주석처리


파이토치 버전이 업그레이드 됨

다운그레이 방법
pip uninstall python-multipart

pip install python-multipart==0.0.12


'''