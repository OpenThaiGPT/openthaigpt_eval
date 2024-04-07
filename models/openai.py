import requests

_model_name = None
_api_key = None

#Ref: https://platform.openai.com/docs/api-reference
def init(model_name, api_key):
    global _model_name
    global _api_key
    _model_name = model_name
    _api_key = api_key

#Ref: https://platform.openai.com/docs/api-reference
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
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 512
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    resp = response.json()
    print(resp)
    
    return resp['choices'][0]['message']['content']