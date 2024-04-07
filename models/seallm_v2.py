from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Ensure CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Ref: https://huggingface.co/SeaLLMs/SeaLLM-7B-v2
def init(model_name, model_path=None):
    global tokenizer
    global model
    tokenizer = AutoTokenizer.from_pretrained(model_path or model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path or model_name, trust_remote_code=True, torch_dtype=torch.float16)

    # Move model to CUDA device
    model.to(device)

# Ref: https://huggingface.co/SeaLLMs/SeaLLM-7B-v2
def inference(prompt):
    messages = [
        {"role": "system", "content": 'You are a helpful assistant'},
        {"role": "user", "content": prompt}
    ]
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    # print(tokenizer.convert_ids_to_tokens(encodeds[0]))
    # ['<s>', '▁<', '|', 'im', '_', 'start', '|', '>', 'system', '<0x0A>', 'You', '▁are', '▁a', '▁helpful', '▁assistant', '.', '</s>', '▁<', '|', 'im', '_', 'start', '|', '>', 'user', '<0x0A>', 'Hello', '▁world', '</s>', '▁<', '|', 'im', '_', 'start', '|', '>', 'ass', 'istant', '<0x0A>', 'Hi', '▁there', ',', '▁how', '▁can', '▁I', '▁help', '▁you', '▁today', '?', '</s>', '▁<', '|', 'im', '_', 'start', '|', '>', 'user', '<0x0A>', 'Ex', 'plain', '▁general', '▁rel', 'ativity', '▁in', '▁details', '.', '</s>', '▁<', '|', 'im', '_', 'start', '|', '>', 'ass', 'istant', '<0x0A>']

    model_inputs = encodeds.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.pad_token_id)
    decoded = tokenizer.batch_decode(generated_ids)
    return decoded[0].strip()