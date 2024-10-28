import os
import subprocess
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio

# 모델 사전 로드
preload_models()

# 변환할 텍스트
text_prompt = "안녕하세요, 이 약은 비타민과 같이 먹으면 안됩니다."

# 텍스트를 오디오로 변환
audio_array = generate_audio(text_prompt)

# 오디오 파일로 저장
output_file = "bark_test.wav"
write_wav(output_file, SAMPLE_RATE, audio_array)
print(f"오디오 파일이 생성되었습니다: {os.path.abspath(output_file)}")

# 주피터 노트북이나 IPython 환경에서 오디오 재생
Audio(audio_array, rate=SAMPLE_RATE)

# MacOS에서 오디오 파일 재생
subprocess.run(["afplay", output_file])

# 파일 재생 후 삭제
if os.path.exists(output_file):
    os.remove(output_file)
    print(f"오디오 파일이 삭제되었습니다: {output_file}")
else:
    print("파일이 이미 삭제되었거나 존재하지 않습니다.")
