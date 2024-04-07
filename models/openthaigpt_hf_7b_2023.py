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

#Ref: https://colab.research.google.com/drive/1NkmAJHItpqu34Tur9wCFc97A6JzKR8xo#scrollTo=lsOjziA3Dppt
def inference(prompt):
    llama_prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer.encode(llama_prompt, return_tensors="pt")
    inputs = inputs.to(device)
    outputs = model.generate(inputs, max_length=512, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)