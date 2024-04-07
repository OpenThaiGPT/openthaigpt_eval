from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Ensure CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Ref: https://huggingface.co/SeaLLMs/SeaLLM-7B-v1
def init(model_name, model_path=None):
    global tokenizer
    global model
    tokenizer = AutoTokenizer.from_pretrained(model_path or model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path or model_name, trust_remote_code=True, torch_dtype=torch.float16)

    # Move model to CUDA device
    model.to(device)

# Ref: https://huggingface.co/meta-llama/Llama-2-7b
def inference(prompt):
    inputs = tokenizer.encode(chat_multiturn_seq_format(prompt), return_tensors="pt")
    inputs = inputs.to(device)
    outputs = model.generate(inputs, max_length=512, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Ref: https://huggingface.co/SeaLLMs/SeaLLM-7B-v1
BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SYSTEM_PROMPT = """You are a multilingual, helpful, respectful and honest assistant. \
Please always answer as helpfully as possible, while being safe. Your \
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure \
that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
correct. If you don't know the answer to a question, please don't share false information.

As a multilingual assistant, you must respond and follow instructions in the native language of the user by default, unless told otherwise. \
Your response should adapt to the norms and customs of the respective language and culture.
"""

def chat_multiturn_seq_format(
    message: str,
    history: list[tuple[str, str]] = [], 
):
    """
    ```
        <bos>[INST] B_SYS SytemPrompt E_SYS Prompt [/INST] Answer <eos>
        <bos>[INST] Prompt [/INST] Answer <eos>
        <bos>[INST] Prompt [/INST]
    ```
    As the format auto-add <bos>, please turn off add_special_tokens with `tokenizer.add_special_tokens = False`
    Inputs:
      message: the current prompt
      history: list of list indicating previous conversation. [[message1, response1], [message2, response2]]
    Outputs:
      full_prompt: the prompt that should go into the chat model

    e.g:
      full_prompt = chat_multiturn_seq_format("Hello world")
      output = model.generate(tokenizer.encode(full_prompt, add_special_tokens=False), ...)
    """
    text = ''
    for i, (prompt, res) in enumerate(history):
        if i == 0:
            text += f"{BOS_TOKEN}{B_INST} {B_SYS} {SYSTEM_PROMPT} {E_SYS} {prompt} {E_INST}"
        else:
            text += f"{BOS_TOKEN}{B_INST} {prompt}{E_INST}"
        if res is not None:
            text += f" {res} {EOS_TOKEN} "
    if len(history) == 0 or text.strip() == '':
        text = f"{BOS_TOKEN}{B_INST} {B_SYS} {SYSTEM_PROMPT} {E_SYS} {message} {E_INST}"
    else:
        text += f"{BOS_TOKEN}{B_INST} {message} {E_INST}"
    return text
