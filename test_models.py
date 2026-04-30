import httpx
import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv('OPENROUTER_API_KEY')

models = [
    'google/gemma-4-31b-it:free',
    'deepseek/deepseek-r1:free',
    'deepseek/deepseek-chat:free',
    'qwen/qwen3-8b:free'
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