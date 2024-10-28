from gtts import gTTS
import os
import subprocess

# 변환할 텍스트
text = "안녕하세요, Bark TTS 테스트입니다. 목소리가 고정되어 있습니다."

# gTTS로 한국어 음성 생성
tts = gTTS(text=text, lang='ko')
output_file = "gtts_korean_test.wav"
tts.save(output_file)
print(f"오디오 파일이 생성되었습니다: {os.path.abspath(output_file)}")

# MacOS에서 오디오 파일 재생
subprocess.run(["afplay", output_file])

# 파일 재생 후 삭제
if os.path.exists(output_file):
    os.remove(output_file)
    print(f"오디오 파일이 삭제되었습니다: {output_file}")
else:
    print("파일이 이미 삭제되었거나 존재하지 않습니다.")
