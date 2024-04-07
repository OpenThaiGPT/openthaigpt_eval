from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Ensure CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

def init(model_name, model_path=None):
    global tokenizer
    global model
    
    # Ref: https://huggingface.co/pythainlp/wangchanglm-7.5B-sft-enth
    model = AutoModelForCausalLM.from_pretrained(
        model_path or model_name, 
        return_dict=True, 
        # load_in_8bit=False, # Can not run if load_in_8bit and we testing on fp16
        # device_map="auto", # Can not run if device_map="auto"
        torch_dtype=torch.float16, 
        offload_folder="./", 
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path or model_name)
    
    # Move model to CUDA device
    model.to(device)

def inference(prompt):
    # Need to add "\nตอบว่า:" otherwise, the model will just continue writing the question.
    batch = tokenizer(prompt+"\nตอบว่า:", return_tensors="pt")
    batch.to(device)
    
    # Ref: https://huggingface.co/pythainlp/wangchanglm-7.5B-sft-enth
    with torch.cuda.amp.autocast(): 
        output_tokens = model.generate(
            input_ids=batch["input_ids"],
            max_new_tokens=512, # 512
            no_repeat_ngram_size=2,
            
            #oasst k50
            top_k=50,
            top_p=0.95, # 0.95
            typical_p=1.,
            temperature=0.9, # 0.9
            
            # #oasst typical3
            # typical_p = 0.3,
            # temperature = 0.8,
            # repetition_penalty = 1.2,
        )
    response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return response.strip()