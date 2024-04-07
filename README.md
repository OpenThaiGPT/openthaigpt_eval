# OpenThaiGPT - Thai Exams Eval
Kobkrit Viriyayudhakorn (kobkrit@aieat.or.th)

Usage: ``python evaluate.py <model_name> [model_path/api_key]`` for evaluate the model with all exams in exams folder with the given model name.

Available benchmark models:
 - openthaigpt/openthaigpt-1.0.0-beta-7b-chat-ckpt-hf
 - openthaigpt/openthaigpt-1.0.0-beta-13b-chat-hf
 - openthaigpt/openthaigpt-1.0.0-7b-chat
 - openthaigpt/openthaigpt-1.0.0-13b-chat
 - openthaigpt/openthaigpt-1.0.0-70b-chat
 - sail/Sailor-7B-Chat
 - pythainlp/wangchanglm-7.5B-sft-enth
 - aisingapore/sea-lion-7b-instruct
 - SeaLLMs/SeaLLM-7B-v1
 - SeaLLMs/SeaLLM-7B-v2
 - claude-3-opus-20240229
 - claude-3-sonnet-20240229
 - claude-3-haiku-20240307
 - typhoon-instruct
 - gpt-3.5-turbo
 - gpt-4
 - gemini-pro-1.5

Available benchmark datasets:
- A-Level
- TGAT
- TPAT1
- Thai Investment Consultant
- Facebook Belebele Thai
- xcopa_th_200
- xnli2.0_th_200
- Thai ONET M3
- Thai ONET M6

## Exams Details
https://docs.google.com/spreadsheets/d/1ZtP5Jkx0IvCWNPQhMKitZszGnLKqvEDEf0OKdmQiXjA/edit#gid=1181424412

## Creating a Conda environment
```
conda create --name otg-exam-eval python=3.11
```

## Activating the Conda environment
```
conda activate otg-exam-eval
```

## Installing the required packages
```
pip install -r requirements.txt
```

## Run Evaluation
```
./run.sh
```


