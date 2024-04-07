from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Ensure CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

def init(model_name, model_path=None):
    global tokenizer
    global model
    tokenizer = AutoTokenizer.from_pretrained(model_path or model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path or model_name, trust_remote_code=True, torch_dtype=torch.float16)

    # Move model to CUDA device
    model.to(device)

def inference(prompt):
    # Ref: https://huggingface.co/sail/Sailor-7B-Chat
    messages = [
        {"role": "system", "content": 'You are a helpful assistant'},
        {"role": "question", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    input_ids = model_inputs.input_ids.to(device)

    generated_ids = model.generate(
        input_ids,
        max_new_tokens=512,
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()