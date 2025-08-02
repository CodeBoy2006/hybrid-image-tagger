import base64
import io
import requests
from typing import Optional
from PIL import Image
from . import config
from .utils import logger
from pathlib import Path
from . import config

def compress_image_for_api(image: Image.Image) -> str:
    buffer = io.BytesIO()
    current_quality = 85
    image.save(buffer, format='JPEG', quality=current_quality, optimize=True)
    while buffer.tell() > config.TARGET_MAX_FILE_SIZE and current_quality > 30:
        current_quality -= 10
        buffer.seek(0); buffer.truncate(0)
        image.save(buffer, format='JPEG', quality=current_quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def generate_llm_tags(image: Image.Image, api_key: str, api_url: str, model: str, prompt: str) -> Optional[str]:
    if not api_key: raise ValueError("API Key is required for LLM tagging.")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    base64_image = compress_image_for_api(image)
    payload = {"model": model, "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}], "max_tokens": config.DEFAULT_MAX_TOKENS, "temperature": config.DEFAULT_TEMPERATURE}
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip().replace('\n', ', ')
    except requests.exceptions.RequestException as e: return f"Error: API request failed. {e}"
    except (KeyError, IndexError, TypeError) as e: return f"Error: Invalid API response. {e}"

def merge_tags_intelligent(danbooru_tags: str, natural_tags: str, api_key: str, api_url: str, model: str) -> str:
    try:
        merge_prompt = load_prompt('dual_channel').format(danbooru_tags=danbooru_tags, natural_tags=natural_tags)
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": [{"role": "user", "content": merge_prompt}], "max_tokens": config.DEFAULT_MAX_TOKENS, "temperature": 0.3}
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip().replace('\n', ', ')
    except Exception as e:
        logger.warning(f"Intelligent merging failed: {e}. Falling back to simple concatenation.")
        return f"{danbooru_tags}, {natural_tags}"

def load_prompt(prompt_name: str) -> str:
    script_dir = Path(__file__).parent.parent
    prompt_file = script_dir / config.PROMPT_FILES[prompt_name]
    if not prompt_file.exists(): raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    with open(prompt_file, 'r', encoding='utf-8') as f: return f.read().strip()