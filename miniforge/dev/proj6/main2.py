from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("ai-forever/mGPT-13B")
model = AutoModelForCausalLM.from_pretrained("ai-forever/mGPT-13B")
inputs = tokenizer("Hello", return_tensors="pt")
output = model.generate(inputs["input_ids"], max_length=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
