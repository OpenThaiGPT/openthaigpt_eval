import requests

_model_name = None
_api_key = None

#Ref: https://docs.anthropic.com/claude/reference/getting-started-with-the-api
def init(model_name, api_key):
    global _model_name
    global _api_key
    _model_name = model_name
    _api_key = api_key

#Ref: https://docs.anthropic.com/claude/reference/getting-started-with-the-api
def inference(prompt):
    global _model_name
    global _api_key
    print(_model_name, _api_key)
    headers = {
        "x-api-key": _api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    data = {
        "model": _model_name,
        "max_tokens": 512,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data)
    resp = response.json()
    print(resp)
    
    return resp['content'][0]['text']