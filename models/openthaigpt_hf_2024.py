from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Ensure CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

#Ref: https://huggingface.co/openthaigpt/openthaigpt-1.0.0-7b-chat
def init(model_name, model_path=None):
    global tokenizer
    global model
    tokenizer = AutoTokenizer.from_pretrained(model_path or model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path or model_name, trust_remote_code=True, torch_dtype=torch.float16)

    # Move model to CUDA device
    model.to(device)

#Ref: https://huggingface.co/openthaigpt/openthaigpt-1.0.0-7b-chat
def inference(prompt):
    llama_prompt = f"<s>[INST] <<SYS>>\nYou are a student sitting in an exam. Answer the question with the correct choice and explain the reasoning คุณคือนักเรียนที่กำลังทำข้อสอบ จงตอบคำถามโดยเลือกช้อยส์ที่ถูกต้องพร้อมทั้งอธิบายเหตุผล<</SYS>>\n\n{prompt} [/INST]"
    inputs = tokenizer.encode(llama_prompt, return_tensors="pt")
    inputs = inputs.to(device)
    outputs = model.generate(inputs, max_length=512, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)