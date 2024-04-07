from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Ensure CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

#Ref: https://huggingface.co/aisingapore/sea-lion-7b-instruct
def init(model_name, model_path=None):
    global tokenizer
    global model
    tokenizer = AutoTokenizer.from_pretrained(model_path or model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path or model_name, trust_remote_code=True, torch_dtype=torch.float16)

    # Move model to CUDA device
    model.to(device)

#Ref: https://huggingface.co/aisingapore/sea-lion-7b-instruct
def inference(prompt):
    tokens = tokenizer(f"### USER:\n{prompt}\n\n### RESPONSE:\n", return_tensors="pt")
    # Move tokens to CUDA device
    tokens = tokens.to(device)
    output = model.generate(tokens["input_ids"], max_new_tokens=20, eos_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.split("### RESPONSE:")[-1].strip()