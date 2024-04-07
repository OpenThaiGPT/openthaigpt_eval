REGION = "asia-southeast1"
PROJECT_ID = "gemini-4-417315"

import requests

_model_name = None
_api_key = None

#Ref: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini#model_versions
def init(model_name, api_key):
    global _model_name
    global _api_key
    _model_name = model_name
    _api_key = api_key

#Ref: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini#model_versions
def inference(prompt):
    global _model_name
    global _api_key
    print(_model_name, _api_key)
    headers = {
        "Content-Type": "application/json",
    }

    data = {"contents":[{"parts":[{"text":prompt}]}]}

    response = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={_api_key}", headers=headers, json=data)
    resp = response.json()
    print(resp)
    
    return resp['candidates'][0]['content']['parts'][0]['text']