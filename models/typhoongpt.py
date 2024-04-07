import requests

_model_name = None
_api_key = None

#Ref: https://docs.opentyphoon.ai/
def init(model_name, api_key):
    global _model_name
    global _api_key
    _model_name = model_name
    _api_key = api_key

#Ref: https://docs.opentyphoon.ai/
def inference(prompt):
    global _model_name
    global _api_key
    print(_model_name, _api_key)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_api_key}"
    }

    data = {
        "model": _model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. You must answer only in Thai."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 500,
        "temperature": 0.7,
        "top_p": 1,
        "top_k": 50,
        "repetition_penalty": 1.15,
        "stream": False
    }
    
    response = requests.post("https://api.opentyphoon.ai/v1/chat/completions", headers=headers, json=data)
    resp = response.json()
    print(resp)
    
    return resp['choices'][0]['message']['content']