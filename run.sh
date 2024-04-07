#!/bin/bash

## Opensource model: If you did not load model yet, automatically model download.

# python evaluate.py sail/Sailor-7B-Chat
# python evaluate.py pythainlp/wangchanglm-7.5B-sft-enth
# python evaluate.py aisingapore/sea-lion-7b-instruct

## Opensource model: If you already loaded model, specify model_path.

# python evaluate.py openthaigpt/openthaigpt-1.0.0-beta-7b-chat-ckpt-hf /home/iapp/Storage/openthaigpt_openthaigpt-1.0.0-beta-7b-chat-ckpt-hf
# python evaluate.py openthaigpt/openthaigpt-1.0.0-beta-13b-chat-hf /home/iapp/Storage/openthaigpt_openthaigpt-1.0.0-beta-13b-chat-hf
python evaluate.py openthaigpt/openthaigpt-1.0.0-7b-chat /home/iapp/Storage/openthaigpt_openthaigpt-1.0.0-7b-chat
# python evaluate.py openthaigpt/openthaigpt-1.0.0-13b-chat /home/iapp/Storage/openthaigpt_openthaigpt-1.0.0-13b-chat
# python evaluate.py openthaigpt/openthaigpt-1.0.0-70b-chat /home/iapp/Storage/openthaigpt_openthaigpt-1.0.0-70b-chat
# python evaluate.py sail/Sailor-7B-Chat /home/iapp/Storage/sail_Sailor-7B-Chat
# python evaluate.py pythainlp/wangchanglm-7.5B-sft-enth /home/iapp/Storage/pythainlp_wangchanglm-7.5B-sft-enth
# python evaluate.py aisingapore/sea-lion-7b-instruct /home/iapp/Storage/aisingapore_sea-lion-7b-instruct
# python evaluate.py SeaLLMs/SeaLLM-7B-v1 /home/iapp/Storage/SeaLLMs_SeaLLM-7B-v1
# python evaluate.py SeaLLMs/SeaLLM-7B-v2 /home/iapp/Storage/SeaLLMs_SeaLLM-7B-v2

## For api-base models, specify model name and its api key.

# python evaluate.py claude-3-opus-20240229 {CLAUDE_API_KEY}
# python evaluate.py claude-3-sonnet-20240229 {CLAUDE_API_KEY}
# python evaluate.py claude-3-haiku-20240307 {CLAUDE_API_KEY}
# python evaluate.py gpt-3.5-turbo {OPENAI_API_KEY}
# python evaluate.py gpt-4 {OPENAI_API_KEY}
# python evaluate.py typhoon-instruct {TYPHOON_API_KEY}
# python evaluate.py gemini-pro-1.5 {VERTEX_AI_API_KEY}