from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Ensure CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

#Ref: https://huggingface.co/openthaigpt/openthaigpt-1.0.0-beta-13b-chat-hf
def init(model_name, model_path=None):
    global tokenizer
    global model
    tokenizer = AutoTokenizer.from_pretrained(model_path or model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path or model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map='auto')

    # Move model to CUDA device
    # model.to(device)

#Ref: https://huggingface.co/openthaigpt/openthaigpt-1.0.0-beta-13b-chat-hf
def inference(prompt):
    llama_prompt = f"<s>[INST] <<SYS>>\nYou are a question answering assistant. Answer the question as truthful and helpful as possible คุณคือผู้ช่วยตอบคำถาม จงตอบคำถามอย่างถูกต้องและมีประโยชน์ที่สุด<</SYS>>\n\n{prompt} [/INST]"
    inputs = tokenizer.encode(llama_prompt, return_tensors="pt")
    # inputs = inputs.to(device)
    outputs = model.generate(inputs, max_length=512, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)