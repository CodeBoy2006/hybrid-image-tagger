#!/usr/bin/env python3
import argparse
import base64
import concurrent.futures
import io
import json
import os
import random
import requests
import sys
import time
import traceback
import zipfile
import tempfile
import re
from pathlib import Path
from typing import List, Optional, Dict, Tuple, NamedTuple
from datetime import datetime

# Third-party libraries
import gradio as gr
import numpy as np
import onnxruntime as ort
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps
from tqdm import tqdm

# --- CONSOLE FORMATTING UTILITIES ---

class Colors:
    """ANSI color codes for enhanced console output. This class is now complete."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    
    # Standard colors
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors (FIX: All bright variants are now included)
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background colors
    BG_RED = '\033[41m'

class DebugLogger:
    """Enhanced debug logger with visual formatting"""
    def __init__(self):
        self.start_time = time.time()
        self.session_id = f"TAG_{int(time.time())}"
    def _get_timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]
    def header(self, title: str, char: str = "=", width: int = 80):
        border = char * width; padding = (width - len(title) - 2) // 2; centered_title = f"{' ' * padding}{title}{' ' * padding}"; print(f"\n{Colors.BRIGHT_CYAN}{border}\n{Colors.BOLD}{Colors.BRIGHT_WHITE}{centered_title.ljust(width)}\n{Colors.BRIGHT_CYAN}{border}{Colors.RESET}\n")
    def section(self, title: str):
        print(f"\n{Colors.BRIGHT_CYAN}{'─' * 20} {Colors.BOLD}{title} {'─' * 20}{Colors.RESET}")
    def info(self, message: str, indent: int = 0):
        print(f"{'  ' * indent}{Colors.BRIGHT_CYAN}[{self._get_timestamp()}]{Colors.RESET} {Colors.WHITE}ℹ️  {message}{Colors.RESET}")
    def success(self, message: str, indent: int = 0):
        print(f"{'  ' * indent}{Colors.BRIGHT_GREEN}[{self._get_timestamp()}]{Colors.RESET} {Colors.BRIGHT_GREEN}✅ {message}{Colors.RESET}")
    def warning(self, message: str, indent: int = 0):
        print(f"{'  ' * indent}{Colors.BRIGHT_YELLOW}[{self._get_timestamp()}]{Colors.RESET} {Colors.BRIGHT_YELLOW}⚠️  {message}{Colors.RESET}")
    def error(self, message: str, indent: int = 0):
        print(f"{'  ' * indent}{Colors.BRIGHT_RED}[{self._get_timestamp()}]{Colors.RESET} {Colors.BRIGHT_RED}❌ {message}{Colors.RESET}")
    def debug(self, message: str, indent: int = 0):
        print(f"{'  ' * indent}{Colors.DIM}[{self._get_timestamp()}]{Colors.RESET} {Colors.DIM}🔍 {message}{Colors.RESET}")
    def process_start(self, process_name: str):
        print(f"{Colors.BRIGHT_MAGENTA}[{self._get_timestamp()}]{Colors.RESET} {Colors.BRIGHT_MAGENTA}🚀 Starting: {Colors.BOLD}{process_name}{Colors.RESET}")
    def process_end(self, process_name: str, duration: float, status: str = "completed"):
        status_emoji = "✅" if status == "completed" else "❌"; print(f"{Colors.BRIGHT_MAGENTA}[{self._get_timestamp()}]{Colors.RESET} {Colors.BRIGHT_MAGENTA}{status_emoji} Finished: {Colors.BOLD}{process_name}{Colors.RESET} ({duration:.2f}s)")
    def metric(self, name: str, value: str, unit: str = "", indent: int = 1):
        print(f"{'  ' * indent}{Colors.BRIGHT_CYAN}📊 {Colors.BOLD}{name}:{Colors.RESET} {Colors.BRIGHT_WHITE}{value}{unit}{Colors.RESET}")

# Initialize global logger
logger = DebugLogger()

# --- CONFIGURATION ---
DEFAULT_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_MAX_TOKENS = 400
DEFAULT_TEMPERATURE = 0.4
WD_MODEL_REPO = "SmilingWolf/wd-vit-large-tagger-v3"
WD_MODEL_FILENAME = "model.onnx"
WD_TAGS_FILENAME = "selected_tags.csv"
WD_MODEL_IMG_SIZE = 448
DEFAULT_THRESHOLD = 0.35
TARGET_MAX_FILE_SIZE = 2 * 1024 * 1024  # 2MB
PROMPT_FILES = {'default': 'default_prompt.txt', 'dual_channel': 'dual_channel_prompt.txt'}

# --- RESULT STRUCTURES ---
class TaggingResult(NamedTuple):
    wd_tags: Optional[str]
    llm_tags: Optional[str]
    merged_tags: Optional[str]
    error: Optional[str]

class PostProcessSettings(NamedTuple):
    """Settings for post-processing tags"""
    replace_underscores: bool = False
    escape_brackets: bool = False
    normalize_spaces: bool = True
    remove_duplicates: bool = True
    sort_alphabetically: bool = False
    trigger_word_prefix: str = ""
    trigger_word_suffix: str = ""
    custom_replacements: Dict[str, str] = {}
    max_tags: int = 0  # 0 means no limit
    min_tag_length: int = 1

# --- POST-PROCESSING FUNCTIONS ---

def apply_post_processing(tags: str, settings: PostProcessSettings) -> str:
    """Apply comprehensive post-processing to tag strings"""
    if not tags or not tags.strip():
        return ""
    
    logger.debug(f"Applying post-processing: {len(settings.__dict__)} settings active")
    
    # Split tags and clean initial whitespace
    tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
    
    # Apply individual tag transformations
    processed_tags = []
    for tag in tag_list:
        # Apply custom replacements first (highest priority)
        for old_text, new_text in settings.custom_replacements.items():
            if old_text:  # Only apply non-empty replacements
                tag = tag.replace(old_text, new_text)
        
        # Replace underscores with spaces
        if settings.replace_underscores:
            tag = tag.replace('_', ' ')
        
        # Escape brackets for some training frameworks
        if settings.escape_brackets:
            tag = tag.replace('(', '\\(').replace(')', '\\)')
            tag = tag.replace('[', '\\[').replace(']', '\\]')
            tag = tag.replace('{', '\\{').replace('}', '\\}')
        
        # Normalize multiple spaces to single spaces
        if settings.normalize_spaces:
            tag = re.sub(r'\s+', ' ', tag).strip()
        
        # Filter by minimum length
        if len(tag) >= settings.min_tag_length:
            processed_tags.append(tag)
    
    # Remove duplicates while preserving order
    if settings.remove_duplicates:
        seen = set()
        unique_tags = []
        for tag in processed_tags:
            tag_lower = tag.lower()
            if tag_lower not in seen:
                seen.add(tag_lower)
                unique_tags.append(tag)
        processed_tags = unique_tags
    
    # Sort alphabetically if requested
    if settings.sort_alphabetically:
        processed_tags.sort(key=str.lower)
    
    # Apply tag limit
    if settings.max_tags > 0:
        processed_tags = processed_tags[:settings.max_tags]
    
    # Join tags back together
    result = ', '.join(processed_tags)
    
    # Add trigger words
    prefix = settings.trigger_word_prefix.strip()
    suffix = settings.trigger_word_suffix.strip()
    
    if prefix and suffix:
        result = f"{prefix}, {result}, {suffix}"
    elif prefix:
        result = f"{prefix}, {result}"
    elif suffix:
        result = f"{result}, {suffix}"
    
    logger.debug(f"Post-processing complete: {len(processed_tags)} tags processed")
    return result

def parse_custom_replacements(replacement_text: str) -> Dict[str, str]:
    """Parse custom replacement rules from text input"""
    replacements = {}
    if not replacement_text.strip():
        return replacements
    
    lines = replacement_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):  # Skip empty lines and comments
            continue
        
        if ' -> ' in line:
            old_text, new_text = line.split(' -> ', 1)
            replacements[old_text.strip()] = new_text.strip()
        elif '=' in line:
            old_text, new_text = line.split('=', 1)
            replacements[old_text.strip()] = new_text.strip()
    
    return replacements

# --- PROMPT LOADING ---
def load_prompt(prompt_name: str) -> str:
    script_dir = Path(__file__).parent
    prompt_file = script_dir / PROMPT_FILES[prompt_name]
    if not prompt_file.exists(): raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    with open(prompt_file, 'r', encoding='utf-8') as f: return f.read().strip()

# --- WD TAGGER ENGINE ---
class WDTagger:
    def __init__(self, model_path: Path, tags_path: Path):
        self.model_path, self.tags_path = model_path, tags_path
        logger.section("WD TAGGER INITIALIZATION")
        self.ort_session = ort.InferenceSession(str(self.model_path))
        self.tag_df = pd.read_csv(self.tags_path)
        logger.success("WD Tagger initialized successfully.")

    @staticmethod
    def download_model_and_tags(cache_dir="wd_model_cache") -> Tuple[Path, Path]:
        logger.section("WD TAGGER MODEL ACQUISITION")
        logger.info("Ensuring WD Tagger model files are available...")
        
        repo_id = WD_MODEL_REPO
        filenames = [WD_MODEL_FILENAME, WD_TAGS_FILENAME]
        file_paths = []

        for filename in filenames:
            try:
                logger.info(f"Attempting to load '{filename}' from local cache...")
                local_path_str = hf_hub_download(
                    repo_id=repo_id, filename=filename, cache_dir=cache_dir, local_files_only=True
                )
                file_paths.append(Path(local_path_str))
                logger.success(f"Successfully loaded '{filename}' from local cache.")
            except FileNotFoundError:
                logger.warning(f"'{filename}' not found in local cache. Attempting to download from Hugging Face Hub...")
                try:
                    online_path_str = hf_hub_download(
                        repo_id=repo_id, filename=filename, cache_dir=cache_dir, local_files_only=False
                    )
                    file_paths.append(Path(online_path_str))
                    logger.success(f"Successfully downloaded and cached '{filename}'.")
                except Exception as e:
                    logger.error(f"Fatal: Failed to download '{filename}'. Error: {e}")
                    raise e
            except Exception as e:
                logger.error(f"Fatal: An unexpected error occurred while accessing cache for '{filename}'. Error: {e}")
                raise e

        model_path, tags_path = file_paths[0], file_paths[1]
        logger.success(f"WD model files are ready.\n  Model: {model_path}\n  Tags: {tags_path}")
        return model_path, tags_path
        
    def predict(self, image: Image.Image, threshold: float, hide_ratings: bool, char_first: bool, no_underscore: bool) -> str:
        _, height, width, _ = self.ort_session.get_inputs()[0].shape
        ratio = float(width) / float(height)
        new_height = int(ratio * WD_MODEL_IMG_SIZE)
        processed_image = image.resize((WD_MODEL_IMG_SIZE, new_height), Image.Resampling.BICUBIC)
        image_array = np.asarray(processed_image, dtype=np.float32)[:, :, ::-1]
        image_array = np.expand_dims(image_array, 0)
        input_name = self.ort_session.get_inputs()[0].name
        label_name = self.ort_session.get_outputs()[0].name
        preds = self.ort_session.run([label_name], {input_name: image_array})[0]
        result_df = self.tag_df.copy()
        result_df['probability'] = preds[0]
        filtered_df = result_df[result_df['probability'] > threshold]
        rating_tags = filtered_df[filtered_df['category'] == 1]['name'].tolist()
        char_tags = filtered_df[filtered_df['category'] == 3]['name'].tolist()
        general_tags = filtered_df[filtered_df['category'].isin([0, 4])]['name'].tolist()
        final_tags = (char_tags + general_tags) if char_first else (general_tags + char_tags)
        if not hide_ratings: final_tags.extend([r.replace("rating:", "") for r in rating_tags])
        if no_underscore: final_tags = [t.replace('_', ' ') for t in final_tags]
        return ", ".join(final_tags)

# --- LLM TAGGER ENGINE ---
def compress_image_for_api(image: Image.Image) -> str:
    buffer = io.BytesIO()
    current_quality = 85
    image.save(buffer, format='JPEG', quality=current_quality, optimize=True)
    while buffer.tell() > TARGET_MAX_FILE_SIZE and current_quality > 30:
        current_quality -= 10
        buffer.seek(0); buffer.truncate(0)
        image.save(buffer, format='JPEG', quality=current_quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def generate_llm_tags(image: Image.Image, api_key: str, api_url: str, model: str, prompt: str) -> Optional[str]:
    if not api_key: raise ValueError("API Key is required for LLM tagging.")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    base64_image = compress_image_for_api(image)
    payload = {"model": model, "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}], "max_tokens": DEFAULT_MAX_TOKENS, "temperature": DEFAULT_TEMPERATURE}
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
        payload = {"model": model, "messages": [{"role": "user", "content": merge_prompt}], "max_tokens": DEFAULT_MAX_TOKENS, "temperature": 0.3}
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip().replace('\n', ', ')
    except Exception as e:
        logger.warning(f"Intelligent merging failed: {e}. Falling back to simple concatenation.")
        return f"{danbooru_tags}, {natural_tags}"

# --- DUAL-CHANNEL PROCESSING LOGIC ---
def process_dual_channel_quick(image: Image.Image, filename: str, wd_settings: dict, llm_settings: dict, post_settings: PostProcessSettings) -> TaggingResult:
    """Parallel Strategy: Both taggers run concurrently for a single image."""
    logger.process_start(f"Dual Channel (Quick) for {filename}")
    start_time = time.time()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            wd_future = executor.submit(wd_tagger.predict, image.copy(), **wd_settings)
            llm_future = executor.submit(generate_llm_tags, image.copy(), **llm_settings)
            wd_tags = wd_future.result()
            llm_tags = llm_future.result()
        
        # Apply post-processing before merging
        wd_tags_processed = apply_post_processing(wd_tags, post_settings)
        llm_tags_processed = apply_post_processing(llm_tags, post_settings)
        
        merged_tags = merge_tags_intelligent(wd_tags_processed, llm_tags_processed, llm_settings['api_key'], llm_settings['api_url'], llm_settings['model'])
        merged_tags_final = apply_post_processing(merged_tags, post_settings)
        
        logger.process_end(f"Dual Channel (Quick) for {filename}", time.time() - start_time)
        return TaggingResult(wd_tags_processed, llm_tags_processed, merged_tags_final, None)
    except Exception as e:
        logger.process_end(f"Dual Channel (Quick) for {filename}", time.time() - start_time, "failed")
        return TaggingResult(None, None, None, str(e))

def process_dual_channel_standard(image: Image.Image, filename: str, wd_settings: dict, llm_settings: dict, post_settings: PostProcessSettings) -> TaggingResult:
    """Queue Strategy: Taggers run sequentially for a single image."""
    logger.process_start(f"Dual Channel (Standard) for {filename}")
    start_time = time.time()
    try:
        wd_tags = wd_tagger.predict(image.copy(), **wd_settings)
        llm_tags = generate_llm_tags(image.copy(), **llm_settings)
        
        # Apply post-processing before merging
        wd_tags_processed = apply_post_processing(wd_tags, post_settings)
        llm_tags_processed = apply_post_processing(llm_tags, post_settings)
        
        merged_tags = merge_tags_intelligent(wd_tags_processed, llm_tags_processed, llm_settings['api_key'], llm_settings['api_url'], llm_settings['model'])
        merged_tags_final = apply_post_processing(merged_tags, post_settings)
        
        logger.process_end(f"Dual Channel (Standard) for {filename}", time.time() - start_time)
        return TaggingResult(wd_tags_processed, llm_tags_processed, merged_tags_final, None)
    except Exception as e:
        logger.process_end(f"Dual Channel (Standard) for {filename}", time.time() - start_time, "failed")
        return TaggingResult(None, None, None, str(e))

def process_dual_channel_detailed(image: Image.Image, filename: str, wd_settings: dict, llm_settings: dict, post_settings: PostProcessSettings, output_dir: Path) -> TaggingResult:
    """Parallel Strategy + Full Data Dump for analysis."""
    logger.process_start(f"Dual Channel (Detailed) for {filename}")
    start_time = time.time()
    try:
        res = process_dual_channel_quick(image, filename, wd_settings, llm_settings, post_settings)
        if res.error: return res
        
        # Save both raw and processed versions
        (output_dir / f"{filename}_wd.txt").write_text(res.wd_tags or "", encoding='utf-8')
        (output_dir / f"{filename}_llm.txt").write_text(res.llm_tags or "", encoding='utf-8')
        (output_dir / f"{filename}_merged.txt").write_text(res.merged_tags or "", encoding='utf-8')
        
        logger.success(f"Saved detailed files for {filename}", indent=1)
        logger.process_end(f"Dual Channel (Detailed) for {filename}", time.time() - start_time)
        return res
    except Exception as e:
        logger.process_end(f"Dual Channel (Detailed) for {filename}", time.time() - start_time, "failed")
        return TaggingResult(None, None, None, str(e))

# --- MAIN PROCESSING FUNCTION ---
def process_images_in_parallel(
    files, mode, dual_mode, max_workers, wd_threshold, hide_ratings, char_first, no_underscore, 
    llm_api_key, llm_api_url, llm_model,
    # Post-processing parameters
    pp_replace_underscores, pp_escape_brackets, pp_normalize_spaces, pp_remove_duplicates, 
    pp_sort_alphabetically, pp_trigger_prefix, pp_trigger_suffix, pp_custom_replacements,
    pp_max_tags, pp_min_tag_length,
    progress=gr.Progress(track_tqdm=True)
):
    logger.header("BATCH IMAGE PROCESSING SESSION")
    if not files:
        yield "Please upload one or more images.", None; return
    
    total_files = len(files)
    logger.metric("Total files", str(total_files))
    logger.metric("Processing mode", mode)
    if mode == "Dual Channel": logger.metric("Dual Channel Strategy", dual_mode)
    logger.metric("Batch Concurrency", str(max_workers))
    
    # Setup post-processing settings
    custom_replacements = parse_custom_replacements(pp_custom_replacements)
    post_settings = PostProcessSettings(
        replace_underscores=pp_replace_underscores,
        escape_brackets=pp_escape_brackets,
        normalize_spaces=pp_normalize_spaces,
        remove_duplicates=pp_remove_duplicates,
        sort_alphabetically=pp_sort_alphabetically,
        trigger_word_prefix=pp_trigger_prefix,
        trigger_word_suffix=pp_trigger_suffix,
        custom_replacements=custom_replacements,
        max_tags=int(pp_max_tags) if pp_max_tags > 0 else 0,
        min_tag_length=int(pp_min_tag_length) if pp_min_tag_length > 0 else 1
    )
    
    logger.metric("Post-processing active", str(any([
        pp_replace_underscores, pp_escape_brackets, pp_remove_duplicates, 
        pp_sort_alphabetically, bool(pp_trigger_prefix), bool(pp_trigger_suffix),
        bool(custom_replacements), pp_max_tags > 0, pp_min_tag_length > 1
    ])))
    
    # Reset UI elements at the beginning
    yield "🔄 Initializing processing...", None
    
    wd_settings = {'threshold': wd_threshold, 'hide_ratings': hide_ratings, 'char_first': char_first, 'no_underscore': no_underscore}
    llm_settings = {'api_key': llm_api_key, 'api_url': llm_api_url, 'model': llm_model, 'prompt': load_prompt('default')}
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "results"
        output_dir.mkdir(exist_ok=True)
        
        future_to_file = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(max_workers)) as executor:
            logger.section("SUBMITTING TASKS")
            
            for file_path in files:
                filename = Path(file_path).stem
                try:
                    with Image.open(file_path) as im:
                        image = ImageOps.exif_transpose(im).convert("RGB")
                except Exception as e:
                    logger.error(f"Failed to load image {filename}: {e}"); continue
                
                logger.info(f"Submitting: {filename}")
                if mode == "WD Tagger Only":
                    future = executor.submit(lambda img=image.copy(): TaggingResult(
                        apply_post_processing(wd_tagger.predict(img, **wd_settings), post_settings), 
                        None, None, None
                    ))
                elif mode == "LLM Only":
                    future = executor.submit(lambda img=image.copy(): TaggingResult(
                        None, 
                        apply_post_processing(generate_llm_tags(img, **llm_settings), post_settings), 
                        None, None
                    ))
                elif mode == "Dual Channel":
                    if dual_mode == "Quick":
                        future = executor.submit(process_dual_channel_quick, image.copy(), filename, wd_settings, llm_settings, post_settings)
                    elif dual_mode == "Standard":
                        future = executor.submit(process_dual_channel_standard, image.copy(), filename, wd_settings, llm_settings, post_settings)
                    elif dual_mode == "Detailed":
                        future = executor.submit(process_dual_channel_detailed, image.copy(), filename, wd_settings, llm_settings, post_settings, output_dir)
                future_to_file[future] = filename

            logger.section("PROCESSING RESULTS")
            results = {}
            
            # Use tqdm for progress tracking, which gr.Progress(track_tqdm=True) will automatically pick up
            for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(future_to_file), desc="Processing images"):
                filename = future_to_file[future]
                completed_count = len(results) + 1
                
                try:
                    result = future.result()
                    results[filename] = result
                    
                    if result.error:
                        logger.error(f"Error processing {filename}: {result.error}")
                        (output_dir / f"{filename}_error.txt").write_text(result.error, encoding='utf-8')
                        status_text = f"❌ Error: {filename}"
                    else:
                        final_content = ""
                        if mode == "WD Tagger Only": final_content = result.wd_tags
                        elif mode == "LLM Only": final_content = result.llm_tags
                        elif mode == "Dual Channel" and dual_mode != "Detailed": final_content = result.merged_tags
                        
                        if final_content:
                            (output_dir / f"{filename}.txt").write_text(final_content, encoding='utf-8')
                            logger.success(f"Processed and saved: {filename}.txt")
                            status_text = f"✅ Completed: {filename}"
                        elif mode != "Dual Channel" or (mode == "Dual Channel" and dual_mode != "Detailed"):
                            logger.warning(f"No content generated for {filename}")
                            status_text = f"⚠️ No content for: {filename}"
                        else:
                            status_text = f"✅ Completed: {filename} (detailed files saved)"

                except Exception:
                    tb_str = traceback.format_exc()
                    logger.error(f"CRITICAL ERROR on {filename}:")
                    print(f"\n{Colors.BG_RED}{Colors.BOLD}{Colors.WHITE}--- TRACEBACK for {filename} ---{Colors.RESET}\n{Colors.BRIGHT_RED}{tb_str}{Colors.RESET}\n")
                    results[filename] = TaggingResult(None, None, None, tb_str)
                    (output_dir / f"{filename}_error.txt").write_text(tb_str, encoding='utf-8')
                    status_text = f"💥 Critical error: {filename}"
                
                # Yield updated status text
                yield f"{status_text}\n\n📊 Progress: {completed_count}/{len(future_to_file)} images processed", None

        logger.section("PACKAGING RESULTS")
        yield "📦 Packaging results...", None
        
        zip_path = Path(temp_dir) / "tagged_captions.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for file in output_dir.rglob("*.txt"):
                zf.write(file, file.name)
        
        success_count = sum(1 for r in results.values() if not r.error)
        final_message = f"✅ Processing complete!\n\n📈 Results:\n• {success_count} successfully tagged\n• {total_files - success_count} failed\n• {total_files} total images\n\n📁 Download ready!"
        logger.success(final_message)
        yield final_message, str(zip_path)

# --- GRADIO UI LAYOUT ---
def create_ui():
    with gr.Blocks(theme=gr.themes.Soft()) as ui:
        gr.Markdown("# Hybrid Image Tagger\nA versatile tool for superior image tagging, utilizing WD tagger and VLM with advanced post-processing.")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Upload & Configure")
                image_files = gr.File(label="Upload Images", file_count="multiple", file_types=["image"], type="filepath")
                
                mode_selector = gr.Radio(["WD Tagger Only", "LLM Only", "Dual Channel"], label="Tagging Mode", value="Dual Channel")
                
                with gr.Group(visible=True) as dual_channel_group:
                    dual_mode_selector = gr.Radio(
                        ["Quick", "Standard", "Detailed"], label="Dual Channel Strategy", value="Quick",
                        info="Quick: Parallel execution per image. Standard: Sequential. Detailed: Parallel + saves all intermediate files.")

                max_workers_slider = gr.Number(label="Batch Concurrency", value=4, minimum=1, maximum=16, step=1, info="Number of images to process in parallel.")

                with gr.Accordion("⚙️ Tagger Settings", open=False):
                    with gr.Tab("WD Tagger"):
                        threshold_slider = gr.Slider(minimum=0.1, maximum=1.0, value=DEFAULT_THRESHOLD, step=0.01, label="Confidence Threshold")
                        hide_ratings_checkbox = gr.Checkbox(label="Hide Rating Tags", value=False)
                        char_first_checkbox = gr.Checkbox(label="Character Tags First", value=True)
                        no_underscore_checkbox = gr.Checkbox(label="Remove '_' from tags", value=False)
                    with gr.Tab("LLM Tagger"):
                        llm_api_key_textbox = gr.Textbox(label="API Key", type="password", info="Required for LLM and Dual Channel modes.")
                        llm_api_url_textbox = gr.Textbox(label="API URL", value=DEFAULT_API_URL)
                        llm_model_textbox = gr.Textbox(label="Model", value=DEFAULT_MODEL)

                with gr.Accordion("🔧 Post-Processing Options", open=False):
                    with gr.Tab("Text Formatting"):
                        pp_replace_underscores = gr.Checkbox(
                            label="Replace underscores with spaces", 
                            value=False,
                            info="Convert 'long_hair' to 'long hair'"
                        )
                        pp_escape_brackets = gr.Checkbox(
                            label="Escape brackets", 
                            value=False,
                            info=r"Convert '(detailed)' to '\\(detailed\\)' for training compatibility"
                        )
                        pp_normalize_spaces = gr.Checkbox(
                            label="Normalize spaces", 
                            value=True,
                            info="Remove extra spaces and clean up formatting"
                        )
                        pp_remove_duplicates = gr.Checkbox(
                            label="Remove duplicate tags", 
                            value=True,
                            info="Remove duplicate tags while preserving order"
                        )
                        pp_sort_alphabetically = gr.Checkbox(
                            label="Sort tags alphabetically", 
                            value=False,
                            info="Sort all tags in alphabetical order"
                        )
                    
                    with gr.Tab("Trigger Words"):
                        pp_trigger_prefix = gr.Textbox(
                            label="Prefix trigger word(s)", 
                            value="",
                            placeholder="e.g., masterpiece, high quality",
                            info="Words/phrases to add at the beginning"
                        )
                        pp_trigger_suffix = gr.Textbox(
                            label="Suffix trigger word(s)", 
                            value="",
                            placeholder="e.g., detailed, professional",
                            info="Words/phrases to add at the end"
                        )
                    
                    with gr.Tab("Advanced"):
                        pp_custom_replacements = gr.Textbox(
                            label="Custom replacements",
                            value="",
                            placeholder="rating:safe -> safe\nold_text -> new_text\n# Comments start with #",
                            lines=5,
                            info="Custom text replacements. Format: 'old -> new' or 'old = new', one per line"
                        )
                        with gr.Row():
                            pp_max_tags = gr.Number(
                                label="Max tags limit", 
                                value=0, 
                                minimum=0, 
                                step=1,
                                info="0 = no limit"
                            )
                            pp_min_tag_length = gr.Number(
                                label="Min tag length", 
                                value=1, 
                                minimum=1, 
                                step=1,
                                info="Minimum character length for tags"
                            )

                submit_button = gr.Button("🚀 Generate Tags", variant="primary")
            
            with gr.Column(scale=1):
                gr.Markdown("### 2. Processing Status")
                status_textbox = gr.Textbox(
                    label="Status Log", 
                    value="Ready to process images. Upload files and configure settings, then click 'Generate Tags' to begin.",
                    info="Detailed progress and status information appears here.", 
                    interactive=False,
                    lines=8
                )
                
                gr.Markdown("### 3. Download Results")
                download_link = gr.File(label="📁 Download Tagged Captions (.zip)", interactive=False)
        
        # Event handlers
        mode_selector.change(
            lambda mode: gr.update(visible=(mode == "Dual Channel")), 
            inputs=mode_selector, 
            outputs=dual_channel_group
        )
        
        submit_button.click(
            fn=process_images_in_parallel, 
            inputs=[
                image_files, mode_selector, dual_mode_selector, max_workers_slider, threshold_slider, 
                hide_ratings_checkbox, char_first_checkbox, no_underscore_checkbox, 
                llm_api_key_textbox, llm_api_url_textbox, llm_model_textbox,
                # Post-processing inputs
                pp_replace_underscores, pp_escape_brackets, pp_normalize_spaces, pp_remove_duplicates,
                pp_sort_alphabetically, pp_trigger_prefix, pp_trigger_suffix, pp_custom_replacements,
                pp_max_tags, pp_min_tag_length
            ], 
            outputs=[status_textbox, download_link],
        )
    return ui

if __name__ == "__main__":
    logger.header("HYBRID IMAGE TAGGER", "═", 100)
    wd_tagger = WDTagger(*WDTagger.download_model_and_tags())
    app_ui = create_ui()
    app_ui.launch()