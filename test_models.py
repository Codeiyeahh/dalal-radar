import httpx
import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv('OPENROUTER_API_KEY')

models = [
    'google/gemma-3-27b-it:free',
    'google/gemma-3-12b-it:free', 
    'google/gemma-3-4b-it:free',
    'microsoft/phi-4-reasoning-plus:free',
    'qwen/qwen3-14b:free',
    'qwen/qwen3-30b-a3b:free',
    'mistralai/devstral-small:free',
    'tngtech/deepseek-r1t-chimera:free'
]

for model in models:
    try:
        r = httpx.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {key}',
                'Content-Type': 'application/json'
            },
            json={
                'model': model,
                'messages': [{'role': 'user', 'content': 'hi'}],
                'max_tokens': 5
            },
            timeout=10
        )
        print(f'{model}: {r.status_code}')
    except Exception as e:
        print(f'{model}: ERROR {e}')