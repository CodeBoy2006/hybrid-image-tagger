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
    """ANSI color codes for enhanced console output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    
    # Standard colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'

class DebugLogger:
    """Enhanced debug logger with visual formatting"""
    
    def __init__(self):
        self.start_time = time.time()
        self.session_id = f"TAG_{int(time.time())}"
    
    def _get_timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    def _get_elapsed(self) -> str:
        elapsed = time.time() - self.start_time
        return f"{elapsed:.2f}s"
    
    def header(self, title: str, char: str = "=", width: int = 80):
        """Print a formatted header"""
        border = char * width
        padding = (width - len(title) - 2) // 2
        centered_title = f"{' ' * padding}{title}{' ' * padding}"
        if len(centered_title) < width:
            centered_title += ' '
        
        print(f"\n{Colors.BRIGHT_CYAN}{border}")
        print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}{centered_title}")
        print(f"{Colors.BRIGHT_CYAN}{border}{Colors.RESET}\n")
    
    def section(self, title: str):
        """Print a section header"""
        print(f"\n{Colors.BRIGHT_BLUE}{'─' * 20} {Colors.BOLD}{title} {'─' * 20}{Colors.RESET}")
    
    def info(self, message: str, indent: int = 0):
        """Print an info message"""
        prefix = "  " * indent
        timestamp = self._get_timestamp()
        print(f"{prefix}{Colors.BRIGHT_CYAN}[{timestamp}]{Colors.RESET} {Colors.WHITE}ℹ️  {message}{Colors.RESET}")
    
    def success(self, message: str, indent: int = 0):
        """Print a success message"""
        prefix = "  " * indent
        timestamp = self._get_timestamp()
        print(f"{prefix}{Colors.BRIGHT_GREEN}[{timestamp}]{Colors.RESET} {Colors.BRIGHT_GREEN}✅ {message}{Colors.RESET}")
    
    def warning(self, message: str, indent: int = 0):
        """Print a warning message"""
        prefix = "  " * indent
        timestamp = self._get_timestamp()
        print(f"{prefix}{Colors.BRIGHT_YELLOW}[{timestamp}]{Colors.RESET} {Colors.BRIGHT_YELLOW}⚠️  {message}{Colors.RESET}")
    
    def error(self, message: str, indent: int = 0):
        """Print an error message"""
        prefix = "  " * indent
        timestamp = self._get_timestamp()
        print(f"{prefix}{Colors.BRIGHT_RED}[{timestamp}]{Colors.RESET} {Colors.BRIGHT_RED}❌ {message}{Colors.RESET}")
    
    def debug(self, message: str, indent: int = 0):
        """Print a debug message"""
        prefix = "  " * indent
        timestamp = self._get_timestamp()
        print(f"{prefix}{Colors.DIM}[{timestamp}]{Colors.RESET} {Colors.DIM}🔍 {message}{Colors.RESET}")
    
    def process_start(self, process_name: str, details: str = ""):
        """Print process start message"""
        timestamp = self._get_timestamp()
        print(f"{Colors.BRIGHT_MAGENTA}[{timestamp}]{Colors.RESET} {Colors.BRIGHT_MAGENTA}🚀 Starting: {Colors.BOLD}{process_name}{Colors.RESET}")
        if details:
            print(f"    {Colors.DIM}{details}{Colors.RESET}")
    
    def process_end(self, process_name: str, duration: float = None, status: str = "completed"):
        """Print process end message"""
        timestamp = self._get_timestamp()
        duration_str = f" ({duration:.2f}s)" if duration else ""
        status_emoji = "✅" if status == "completed" else "❌" if status == "failed" else "⚠️"
        print(f"{Colors.BRIGHT_MAGENTA}[{timestamp}]{Colors.RESET} {Colors.BRIGHT_MAGENTA}{status_emoji} Finished: {Colors.BOLD}{process_name}{Colors.RESET}{duration_str}")
    
    def metric(self, name: str, value: str, unit: str = "", indent: int = 1):
        """Print a metric"""
        prefix = "  " * indent
        print(f"{prefix}{Colors.BRIGHT_CYAN}📊 {Colors.BOLD}{name}:{Colors.RESET} {Colors.BRIGHT_WHITE}{value}{unit}{Colors.RESET}")
    
    def file_info(self, filename: str, size: str = "", path: str = "", indent: int = 1):
        """Print file information"""
        prefix = "  " * indent
        size_info = f" ({size})" if size else ""
        path_info = f"\n{prefix}   📁 {Colors.DIM}{path}{Colors.RESET}" if path else ""
        print(f"{prefix}{Colors.BRIGHT_YELLOW}📄 {Colors.BOLD}{filename}{Colors.RESET}{size_info}{path_info}")
    
    def progress_bar(self, current: int, total: int, width: int = 40, prefix: str = "Progress"):
        """Print a custom progress bar"""
        percentage = current / total if total > 0 else 0
        filled = int(width * percentage)
        bar = "█" * filled + "░" * (width - filled)
        percent_str = f"{percentage:.1%}"
        print(f"\r{Colors.BRIGHT_BLUE}{prefix}: {Colors.BRIGHT_GREEN}[{bar}] {percent_str} ({current}/{total}){Colors.RESET}", end="", flush=True)
        if current == total:
            print()  # New line when complete

# Initialize global logger
logger = DebugLogger()

# --- CONFIGURATION ---

# LLM Tagger Defaults
DEFAULT_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_MAX_TOKENS = 400
DEFAULT_TEMPERATURE = 0.4

# WD Tagger Defaults (Updated for v3)
WD_MODEL_REPO = "SmilingWolf/wd-vit-large-tagger-v3"
WD_MODEL_FILENAME = "model.onnx"
WD_TAGS_FILENAME = "selected_tags.csv"
WD_MODEL_IMG_SIZE = 448
DEFAULT_THRESHOLD = 0.35

# Image Processing Defaults
DEFAULT_MAX_DIMS = (2048, 2048)
DEFAULT_QUALITY = 85
TARGET_MAX_FILE_SIZE = 2 * 1024 * 1024  # 2MB

# --- PROMPT FILES ---
PROMPT_FILES = {
    'default': 'default_prompt.txt',
    'dual_channel': 'dual_channel_prompt.txt'
}

# --- RESULT STRUCTURES ---

class TaggingResult(NamedTuple):
    wd_tags: Optional[str]
    llm_tags: Optional[str]
    merged_tags: Optional[str]
    error: Optional[str]

# --- PROMPT LOADING ---

def load_prompt(prompt_name: str) -> str:
    """Load a prompt from a text file in the script directory"""
    logger.process_start(f"Loading prompt: {prompt_name}")
    
    script_dir = Path(__file__).parent
    prompt_file = script_dir / PROMPT_FILES[prompt_name]
    
    logger.file_info(prompt_file.name, path=str(prompt_file.parent))
    
    if not prompt_file.exists():
        logger.error(f"Prompt file not found: {prompt_file}")
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    logger.success(f"Loaded prompt '{prompt_name}' ({len(content)} characters)")
    logger.debug(f"Prompt preview: {content[:100]}..." if len(content) > 100 else content)
    
    return content

# --- WD TAGGER ENGINE ---

class WDTagger:
    def __init__(self, model_path: Path, tags_path: Path):
        logger.section("WD TAGGER INITIALIZATION")
        
        self.model_path = model_path
        self.tags_path = tags_path
        self.ort_session = None
        self.tag_df = None
        
        logger.file_info("Model", path=str(model_path))
        logger.file_info("Tags", path=str(tags_path))
        
        self._load_model_and_tags()

    def _load_model_and_tags(self):
        start_time = time.time()
        logger.process_start("Loading WD Tagger components")
        
        try:
            # Load ONNX model
            logger.info("Loading ONNX Runtime session...", indent=1)
            self.ort_session = ort.InferenceSession(str(self.model_path))
            
            # Get model info
            input_shape = self.ort_session.get_inputs()[0].shape
            output_shape = self.ort_session.get_outputs()[0].shape
            logger.metric("Input Shape", str(input_shape), indent=2)
            logger.metric("Output Shape", str(output_shape), indent=2)
            
            # Load tags CSV
            logger.info("Loading tags database...", indent=1)
            self.tag_df = pd.read_csv(self.tags_path)
            
            logger.metric("Total Tags", str(len(self.tag_df)), indent=2)
            logger.metric("Tag Categories", str(self.tag_df['category'].nunique()), indent=2)
            
            # Show category breakdown
            category_counts = self.tag_df['category'].value_counts().sort_index()
            for cat, count in category_counts.items():
                cat_name = {0: "General", 1: "Rating", 3: "Character", 4: "Meta"}.get(cat, f"Category {cat}")
                logger.metric(f"  {cat_name}", str(count), indent=2)
            
            duration = time.time() - start_time
            logger.process_end("WD Tagger initialization", duration, "completed")
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Failed to load WD Tagger: {e}")
            logger.process_end("WD Tagger initialization", duration, "failed")
            raise

    @staticmethod
    def download_model_and_tags(cache_dir="wd_model_cache") -> Tuple[Path, Path]:
        logger.section("WD TAGGER MODEL DOWNLOAD")
        logger.process_start("Checking/downloading model files")
        
        try:
            logger.info(f"Repository: {WD_MODEL_REPO}")
            logger.info(f"Cache directory: {cache_dir}")
            
            # Download model
            logger.info("Downloading model file...", indent=1)
            model_path_str = hf_hub_download(
                repo_id=WD_MODEL_REPO, 
                filename=WD_MODEL_FILENAME, 
                cache_dir=cache_dir
            )
            model_path = Path(model_path_str)
            
            # Get model file size
            model_size = model_path.stat().st_size / (1024 * 1024)  # MB
            logger.success(f"Model downloaded: {model_path.name} ({model_size:.1f} MB)")
            
            # Download tags
            logger.info("Downloading tags file...", indent=1)
            tags_path_str = hf_hub_download(
                repo_id=WD_MODEL_REPO, 
                filename=WD_TAGS_FILENAME, 
                cache_dir=cache_dir
            )
            tags_path = Path(tags_path_str)
            
            # Get tags file size
            tags_size = tags_path.stat().st_size / 1024  # KB
            logger.success(f"Tags downloaded: {tags_path.name} ({tags_size:.1f} KB)")
            
            logger.file_info("Final model path", path=str(model_path))
            logger.file_info("Final tags path", path=str(tags_path))
            
            return model_path, tags_path
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise

    def predict(self, image: Image.Image, threshold: float, hide_ratings: bool, char_first: bool, no_underscore: bool) -> str:
        start_time = time.time()
        logger.process_start(f"WD Tagger prediction (threshold: {threshold})")
        
        try:
            if not self.ort_session or self.tag_df is None:
                raise RuntimeError("Model is not loaded.")
            
            # Get model dimensions
            _, height, width, _ = self.ort_session.get_inputs()[0].shape
            logger.debug(f"Original image size: {image.size}")
            
            # Preprocess image
            logger.info("Preprocessing image...", indent=1)
            # The image passed here should already be in RGB mode
            ratio = float(width) / float(height)
            new_height = int(ratio * WD_MODEL_IMG_SIZE)
            image = image.resize((WD_MODEL_IMG_SIZE, new_height), Image.Resampling.BICUBIC)
            logger.debug(f"Resized to: {image.size}")
            
            # Convert to array
            image_array = np.asarray(image, dtype=np.float32)
            image_array = image_array[:, :, ::-1]  # RGB to BGR
            image_array = np.expand_dims(image_array, 0)
            logger.debug(f"Array shape: {image_array.shape}")
            
            # Run inference
            logger.info("Running inference...", indent=1)
            input_name = self.ort_session.get_inputs()[0].name
            label_name = self.ort_session.get_outputs()[0].name
            preds = self.ort_session.run([label_name], {input_name: image_array})[0]
            probabilities = preds[0]
            
            # Process results
            logger.info("Processing predictions...", indent=1)
            result_df = self.tag_df.copy()
            result_df['probability'] = probabilities
            filtered_df = result_df[result_df['probability'] > threshold]
            
            logger.metric("Tags above threshold", str(len(filtered_df)), indent=2)
            
            # Categorize tags
            rating_tags = filtered_df[filtered_df['category'] == 1]['name'].tolist()
            char_tags = filtered_df[filtered_df['category'] == 3]['name'].tolist()
            general_tags = filtered_df[filtered_df['category'].isin([0, 4])]['name'].tolist()
            
            logger.metric("Rating tags", str(len(rating_tags)), indent=2)
            logger.metric("Character tags", str(len(char_tags)), indent=2)
            logger.metric("General tags", str(len(general_tags)), indent=2)
            
            # Arrange final tags
            final_tags = []
            if char_first:
                final_tags.extend(char_tags)
                final_tags.extend(general_tags)
                logger.debug("Tag order: Character → General")
            else:
                final_tags.extend(general_tags)
                final_tags.extend(char_tags)
                logger.debug("Tag order: General → Character")
            
            if not hide_ratings:
                final_tags.extend([r.replace("rating:", "") for r in rating_tags])
                logger.debug("Including rating tags")
            else:
                logger.debug("Hiding rating tags")
            
            if no_underscore:
                final_tags = [t.replace('_', ' ') for t in final_tags]
                logger.debug("Removed underscores from tags")
            
            result = ", ".join(final_tags)
            duration = time.time() - start_time
            
            logger.success(f"Generated {len(final_tags)} tags")
            logger.debug(f"Preview: {result[:100]}..." if len(result) > 100 else result)
            logger.process_end("WD Tagger prediction", duration, "completed")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"WD prediction failed: {e}")
            logger.process_end("WD Tagger prediction", duration, "failed")
            raise

# --- LLM TAGGER ENGINE ---

def compress_image_for_api(image: Image.Image) -> str:
    """Compress image for API with debug output"""
    start_time = time.time()
    logger.process_start("Image compression for API")
    
    try:
        # The image is already RGB and EXIF-transposed.
        img_rgb = image 
        logger.debug(f"Original size: {img_rgb.size}")
        
        # Initial compression
        buffer = io.BytesIO()
        img_rgb.save(buffer, format='JPEG', quality=DEFAULT_QUALITY, optimize=True)
        initial_size = buffer.tell()
        logger.metric("Initial compressed size", f"{initial_size / 1024:.1f}", "KB", indent=1)
        
        # Adaptive quality reduction if needed
        current_quality = DEFAULT_QUALITY
        compression_steps = 0
        
        while buffer.tell() > TARGET_MAX_FILE_SIZE and current_quality > 30:
            current_quality -= 10
            compression_steps += 1
            buffer.seek(0)
            buffer.truncate(0)
            img_rgb.save(buffer, format='JPEG', quality=current_quality, optimize=True)
            logger.debug(f"Reduced quality to {current_quality}% → {buffer.tell() / 1024:.1f}KB")
        
        final_size = buffer.tell()
        compression_ratio = (1 - final_size / initial_size) * 100 if initial_size > 0 else 0
        
        # Encode to base64
        base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        duration = time.time() - start_time
        logger.metric("Final size", f"{final_size / 1024:.1f}", "KB", indent=1)
        logger.metric("Compression ratio", f"{compression_ratio:.1f}", "%", indent=1)
        logger.metric("Quality used", f"{current_quality}", "%", indent=1)
        logger.metric("Base64 length", f"{len(base64_data)}", " chars", indent=1)
        
        if compression_steps > 0:
            logger.warning(f"Applied {compression_steps} compression steps to meet size limit")
        
        logger.process_end("Image compression", duration, "completed")
        return base64_data
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Image compression failed: {e}")
        logger.process_end("Image compression", duration, "failed")
        raise

def generate_llm_tags(
    image: Image.Image, api_key: str, api_url: str, model: str, prompt: str, 
    existing_tags: Optional[str] = None
) -> Optional[str]:
    """Generate LLM tags with enhanced debug output"""
    start_time = time.time()
    logger.process_start(f"LLM tagging with {model}")
    
    try:
        if not api_key:
            logger.error("API Key is required for LLM tagging")
            raise ValueError("API Key is required for LLM tagging.")
        
        logger.info("Preparing API request...", indent=1)
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        
        # Compress image
        base64_image = compress_image_for_api(image)
        
        # Prepare prompt
        final_prompt = prompt.format(existing_tags=existing_tags) if existing_tags else prompt
        logger.debug(f"Prompt length: {len(final_prompt)} characters")
        if existing_tags:
            logger.debug(f"Using existing tags: {existing_tags[:100]}...")
        
        # Build payload
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": final_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            "max_tokens": DEFAULT_MAX_TOKENS, 
            "temperature": DEFAULT_TEMPERATURE
        }
        
        logger.metric("API URL", api_url, indent=1)
        logger.metric("Model", model, indent=1)
        logger.metric("Max tokens", str(DEFAULT_MAX_TOKENS), indent=1)
        logger.metric("Temperature", str(DEFAULT_TEMPERATURE), indent=1)
        
        # Make API request
        logger.info("Sending API request...", indent=1)
        response = requests.post(api_url, headers=headers, json=payload, timeout=90)
        
        logger.metric("Response status", str(response.status_code), indent=1)
        logger.metric("Response time", f"{response.elapsed.total_seconds():.2f}", "s", indent=1)
        
        response.raise_for_status()
        
        # Parse response
        response_data = response.json()
        content = response_data['choices'][0]['message']['content']
        
        # Log usage if available
        if 'usage' in response_data:
            usage = response_data['usage']
            logger.metric("Prompt tokens", str(usage.get('prompt_tokens', 'N/A')), indent=1)
            logger.metric("Completion tokens", str(usage.get('completion_tokens', 'N/A')), indent=1)
            logger.metric("Total tokens", str(usage.get('total_tokens', 'N/A')), indent=1)
        
        # Process result
        result = content.strip().replace('\n', ', ')
        duration = time.time() - start_time
        
        logger.success(f"Generated LLM tags ({len(result)} characters)")
        logger.debug(f"Preview: {result[:100]}..." if len(result) > 100 else result)
        logger.process_end("LLM tagging", duration, "completed")
        
        return result
        
    except requests.exceptions.RequestException as e:
        duration = time.time() - start_time
        error_msg = f"API request failed: {e}"
        logger.error(error_msg)
        logger.process_end("LLM tagging", duration, "failed")
        return f"Error: {error_msg}"
        
    except (KeyError, IndexError, TypeError) as e:
        duration = time.time() - start_time
        error_msg = f"Invalid API response: {e}"
        logger.error(error_msg)
        logger.process_end("LLM tagging", duration, "failed")
        return f"Error: {error_msg}"

def merge_tags_intelligent(danbooru_tags: str, natural_tags: str, api_key: str, api_url: str, model: str) -> str:
    """Intelligently merge Danbooru and natural language tags using LLM"""
    start_time = time.time()
    logger.process_start("Intelligent tag merging")
    
    try:
        logger.info("Loading merge prompt...", indent=1)
        merge_prompt = load_prompt('dual_channel').format(
            danbooru_tags=danbooru_tags,
            natural_tags=natural_tags
        )
        
        logger.debug(f"Danbooru tags count: {len(danbooru_tags.split(','))}")
        logger.debug(f"Natural tags preview: {natural_tags[:100]}...")
        
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": merge_prompt}],
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": 0.3  # Lower temperature for more consistent merging
        }
        
        logger.info("Sending merge request...", indent=1)
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        content = response.json()['choices'][0]['message']['content']
        result = content.strip().replace('\n', ', ')
        
        duration = time.time() - start_time
        logger.success(f"Merged tags successfully ({len(result)} characters)")
        logger.debug(f"Merged preview: {result[:100]}..." if len(result) > 100 else result)
        logger.process_end("Intelligent tag merging", duration, "completed")
        
        return result
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Intelligent merging failed: {e}")
        logger.warning("Falling back to simple concatenation")
        logger.process_end("Intelligent tag merging", duration, "failed")
        
        # Fallback: simple concatenation
        fallback_result = f"{danbooru_tags}, {natural_tags}"
        logger.debug(f"Fallback result: {fallback_result[:100]}...")
        return fallback_result

# --- DUAL-CHANNEL PROCESSING ---

def process_dual_channel_quick(
    image: Image.Image, filename: str, wd_settings: dict, llm_settings: dict
) -> TaggingResult:
    """Quick Mode: Both processes run simultaneously, merged once both complete"""
    logger.section(f"DUAL CHANNEL QUICK MODE - {filename}")
    start_time = time.time()
    
    try:
        logger.info("Starting parallel processing...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks simultaneously with their own copies of the image
            logger.info("Submitting WD Tagger task...", indent=1)
            wd_future = executor.submit(
                wd_tagger.predict, image.copy(), # FIX: Pass a copy
                wd_settings['threshold'], wd_settings['hide_ratings'], 
                wd_settings['char_first'], wd_settings['no_underscore']
            )
            
            logger.info("Submitting LLM Tagger task...", indent=1)
            llm_future = executor.submit(
                generate_llm_tags, image.copy(), # FIX: Pass a copy
                llm_settings['api_key'], llm_settings['api_url'], 
                llm_settings['model'], llm_settings['prompt']
            )
            
            logger.info("Waiting for both taggers to complete...", indent=1)
            wd_tags = wd_future.result()
            llm_tags = llm_future.result()
            
            logger.success("Both taggers completed")
            
            # Intelligent merge
            merged_tags = merge_tags_intelligent(
                wd_tags, llm_tags, 
                llm_settings['api_key'], llm_settings['api_url'], llm_settings['model']
            )
            
            duration = time.time() - start_time
            logger.process_end(f"Quick mode processing for {filename}", duration, "completed")
            
            return TaggingResult(wd_tags, llm_tags, merged_tags, None)
            
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Quick mode failed for {filename}: {e}")
        logger.process_end(f"Quick mode processing for {filename}", duration, "failed")
        return TaggingResult(None, None, None, str(e))

def process_dual_channel_standard(
    image: Image.Image, filename: str, wd_settings: dict, llm_settings: dict, output_dir: Path
) -> TaggingResult:
    """Standard Mode: WD first, then LLM, then merge"""
    logger.section(f"DUAL CHANNEL STANDARD MODE - {filename}")
    start_time = time.time()
    
    try:
        # Step 1: WD Tagger
        logger.info("Step 1: Running WD Tagger...")
        wd_tags = wd_tagger.predict(
            image.copy(), # Use a copy for safety, though not strictly parallel
            wd_settings['threshold'], wd_settings['hide_ratings'], 
            wd_settings['char_first'], wd_settings['no_underscore']
        )
        
        # Save WD results
        wd_file = output_dir / f"{filename}_wd.txt"
        wd_file.write_text(wd_tags, encoding='utf-8')
        logger.success(f"Saved WD tags to {wd_file.name}")
        
        # Step 2: LLM Tagger
        logger.info("Step 2: Running LLM Tagger...")
        llm_tags = generate_llm_tags(
            image.copy(), # Use a copy for safety
            llm_settings['api_key'], llm_settings['api_url'], 
            llm_settings['model'], llm_settings['prompt']
        )
        
        # Save LLM results
        llm_file = output_dir / f"{filename}_llm.txt"
        llm_file.write_text(llm_tags, encoding='utf-8')
        logger.success(f"Saved LLM tags to {llm_file.name}")
        
        # Step 3: Intelligent merge
        logger.info("Step 3: Intelligent merging...")
        merged_tags = merge_tags_intelligent(
            wd_tags, llm_tags, 
            llm_settings['api_key'], llm_settings['api_url'], llm_settings['model']
        )
        
        duration = time.time() - start_time
        logger.process_end(f"Standard mode processing for {filename}", duration, "completed")
        
        return TaggingResult(wd_tags, llm_tags, merged_tags, None)
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Standard mode failed for {filename}: {e}")
        logger.process_end(f"Standard mode processing for {filename}", duration, "failed")
        return TaggingResult(None, None, None, str(e))

def process_dual_channel_detailed(
    image: Image.Image, filename: str, wd_settings: dict, llm_settings: dict, output_dir: Path
) -> TaggingResult:
    """Detailed Mode: Save all three sets of tags separately"""
    logger.section(f"DUAL CHANNEL DETAILED MODE - {filename}")
    start_time = time.time()
    
    try:
        logger.info("Processing both taggers in parallel...")
        
        # Process both simultaneously with their own copies of the image
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            wd_future = executor.submit(
                wd_tagger.predict, image.copy(), # FIX: Pass a copy
                wd_settings['threshold'], wd_settings['hide_ratings'], 
                wd_settings['char_first'], wd_settings['no_underscore']
            )
            llm_future = executor.submit(
                generate_llm_tags, image.copy(), # FIX: Pass a copy
                llm_settings['api_key'], llm_settings['api_url'], 
                llm_settings['model'], llm_settings['prompt']
            )
            
            wd_tags = wd_future.result()
            llm_tags = llm_future.result()
        
        # Save individual results
        logger.info("Saving individual tag files...", indent=1)
        wd_file = output_dir / f"{filename}_wd.txt"
        llm_file = output_dir / f"{filename}_llm.txt"
        wd_file.write_text(wd_tags, encoding='utf-8')
        llm_file.write_text(llm_tags, encoding='utf-8')
        
        logger.success(f"Saved WD tags: {wd_file.name}")
        logger.success(f"Saved LLM tags: {llm_file.name}")
        
        # Intelligent merge
        logger.info("Creating intelligent merge...")
        merged_tags = merge_tags_intelligent(
            wd_tags, llm_tags, 
            llm_settings['api_key'], llm_settings['api_url'], llm_settings['model']
        )
        
        # Save merged results
        merged_file = output_dir / f"{filename}_merged.txt"
        merged_file.write_text(merged_tags, encoding='utf-8')
        logger.success(f"Saved merged tags: {merged_file.name}")
        
        duration = time.time() - start_time
        logger.process_end(f"Detailed mode processing for {filename}", duration, "completed")
        
        return TaggingResult(wd_tags, llm_tags, merged_tags, None)
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Detailed mode failed for {filename}: {e}")
        logger.process_end(f"Detailed mode processing for {filename}", duration, "failed")
        return TaggingResult(None, None, None, str(e))

# --- MAIN PROCESSING FUNCTION ---

def process_images_in_parallel(
    files, mode, dual_mode, wd_threshold, hide_ratings, char_first, no_underscore, 
    llm_api_key, llm_api_url, llm_model
):
    logger.header("BATCH IMAGE PROCESSING SESSION")
    session_start = time.time()
    
    if not files:
        logger.warning("No files provided")
        yield "Please upload one or more images.", None
        return
    
    total_files = len(files)
    logger.metric("Total files", str(total_files))
    logger.metric("Processing mode", mode)
    if mode == "Dual Channel":
        logger.metric("Dual channel mode", dual_mode)
    logger.metric("WD threshold", str(wd_threshold))
    
    results = {}
    
    # Load prompts
    try:
        logger.info("Loading prompts...")
        default_prompt = load_prompt('default')
    except FileNotFoundError as e:
        logger.error(f"Prompt loading failed: {e}")
        yield f"Error: {str(e)}", None
        return
    
    # Prepare settings dictionaries
    wd_settings = {
        'threshold': wd_threshold,
        'hide_ratings': hide_ratings,
        'char_first': char_first,
        'no_underscore': no_underscore
    }
    
    llm_settings = {
        'api_key': llm_api_key,
        'api_url': llm_api_url,
        'model': llm_model,
        'prompt': default_prompt
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        output_dir = temp_dir_path / "results"
        output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Created temporary directory: {output_dir}")
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_file = {}
            
            logger.section("SUBMITTING PROCESSING TASKS")
            
            for i, file_path in enumerate(files, 1):
                filename = Path(file_path).stem
                
                try:
                    # FIX: Safely load image into memory for multithreading
                    with Image.open(file_path) as im:
                        im = ImageOps.exif_transpose(im) # Correct orientation
                        image = im.convert("RGB") # Load data and ensure RGB
                except Exception as e:
                    logger.error(f"Failed to load image {filename}: {e}")
                    results[filename] = TaggingResult(None, None, None, f"Image load error: {e}")
                    continue

                logger.info(f"[{i}/{total_files}] Submitting: {filename}")
                logger.debug(f"  Image size: {image.size}")
                logger.debug(f"  Image mode: {image.mode}")
                
                if mode == "WD Tagger Only":
                    future = executor.submit(
                        wd_tagger.predict,
                        image.copy(), wd_threshold, hide_ratings, char_first, no_underscore
                    )
                    # Wrap result in TaggingResult
                    future = executor.submit(
                        lambda img: TaggingResult(img, None, None, None),
                        future.result()
                    ) if future else None

                elif mode == "LLM Only":
                    future = executor.submit(
                        generate_llm_tags,
                        image.copy(), llm_api_key, llm_api_url, llm_model, default_prompt
                    )
                     # Wrap result in TaggingResult
                    future = executor.submit(
                        lambda tags: TaggingResult(None, tags, None, None),
                        future.result()
                    ) if future else None

                elif mode == "Dual Channel":
                    if dual_mode == "Quick":
                        future = executor.submit(process_dual_channel_quick, image, filename, wd_settings, llm_settings)
                    elif dual_mode == "Standard":
                        future = executor.submit(process_dual_channel_standard, image, filename, wd_settings, llm_settings, output_dir)
                    elif dual_mode == "Detailed":
                        future = executor.submit(process_dual_channel_detailed, image, filename, wd_settings, llm_settings, output_dir)
                else:
                    continue
                
                if future:
                    future_to_file[future] = filename
            
            logger.section("PROCESSING RESULTS")
            progress_bar = tqdm(total=len(future_to_file), desc="Processing Images", 
                              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_file):
                filename = future_to_file[future]
                completed += 1
                
                try:
                    result = future.result()
                    results[filename] = result
                    
                    if result.error:
                        logger.error(f"Processing failed for {filename}: {result.error}", indent=1)
                    else:
                        logger.success(f"Successfully processed {filename}")
                        
                        # Log tag counts
                        if result.wd_tags:
                            wd_count = len(result.wd_tags.split(', '))
                            logger.metric("WD tags", str(wd_count), indent=1)
                        if result.llm_tags:
                            llm_count = len(result.llm_tags.split(', ')) if ', ' in result.llm_tags else 1
                            logger.metric("LLM tags", str(llm_count), indent=1)
                        if result.merged_tags:
                            merged_count = len(result.merged_tags.split(', '))
                            logger.metric("Merged tags", str(merged_count), indent=1)
                    
                    # Save primary result file
                    if mode == "WD Tagger Only" and result.wd_tags:
                        output_file = output_dir / f"{filename}.txt"
                        output_file.write_text(result.wd_tags, encoding='utf-8')
                    elif mode == "LLM Only" and result.llm_tags:
                        output_file = output_dir / f"{filename}.txt"
                        output_file.write_text(result.llm_tags, encoding='utf-8')
                    elif mode == "Dual Channel" and dual_mode == "Quick" and result.merged_tags:
                        output_file = output_dir / f"{filename}.txt"
                        output_file.write_text(result.merged_tags, encoding='utf-8')
                    # Standard and Detailed modes already save files during processing
                    
                except Exception as e:
                    tb_str = traceback.format_exc()
                    logger.error(f"UNEXPECTED CRITICAL ERROR while processing '{filename}'")
                    print(f"\n{Colors.BG_RED}{Colors.BOLD}{Colors.WHITE}--- TRACEBACK for {filename} ---{Colors.RESET}")
                    print(f"{Colors.BRIGHT_RED}{tb_str}{Colors.RESET}")
                    print(f"{Colors.BG_RED}{Colors.BOLD}{Colors.WHITE}{'--- END TRACEBACK ---':^30}{Colors.RESET}\n")

                    error_result = TaggingResult(None, None, None, str(e))
                    results[filename] = error_result
                    error_file = output_dir / f"{filename}_error.txt"
                    error_file.write_text(f"Error processing {filename}:\n\n{tb_str}", encoding='utf-8')
                
                progress_bar.update(1)
                yield f"Processing... ({completed}/{total_files})", None
            
            progress_bar.close()
        
        logger.section("CREATING RESULTS PACKAGE")
        
        # Create zip file with all results
        zip_path = temp_dir_path / "tagged_captions.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            file_count = 0
            for file in output_dir.rglob("*.txt"):
                zf.write(file, file.name)
                file_count += 1
                logger.debug(f"Added to zip: {file.name}")
        
        zip_size = zip_path.stat().st_size / 1024  # KB
        logger.success(f"Created results archive: {zip_path.name} ({zip_size:.1f} KB)")
        logger.metric("Files in archive", str(file_count))
        
        # Final statistics
        success_count = sum(1 for r in results.values() if r.error is None)
        error_count = total_files - success_count
        session_duration = time.time() - session_start
        
        logger.section("SESSION SUMMARY")
        logger.metric("Total processed", str(total_files))
        logger.metric("Successful", str(success_count))
        logger.metric("Errors", str(error_count))
        logger.metric("Success rate", f"{(success_count/total_files)*100:.1f}", "%")
        logger.metric("Total time", f"{session_duration:.2f}", "s")
        if total_files > 0:
            logger.metric("Average per image", f"{session_duration/total_files:.2f}", "s")
        
        if error_count > 0:
            logger.warning(f"{error_count} files had errors - check console logs for tracebacks.")
        else:
            logger.success("All files processed successfully! 🎉")
        
        yield f"✅ Processing complete! {success_count}/{total_files} images tagged successfully. Your download is ready.", str(zip_path)

# --- ENHANCED UI LAYOUT ---
def create_ui():
    with gr.Blocks(theme=gr.themes.Soft()) as ui:
        gr.Markdown(
            """
            # 🖼️ Universal AI Image Tagger v2.1
            A versatile tool featuring **dual-channel tagging** that combines the precision of **WD ViT-Large Tagger v3** 
            with the natural language capabilities of modern **Language Models** for superior image tagging results.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Upload Images")
                image_files = gr.File(
                    label="Upload Images", 
                    file_count="multiple", 
                    file_types=["image"], 
                    type="filepath"
                )
                
                gr.Markdown("### 2. Select Tagging Mode")
                mode_selector = gr.Radio(
                    ["WD Tagger Only", "LLM Only", "Dual Channel"], 
                    label="Tagging Mode", 
                    value="Dual Channel"
                )
                
                # Dual Channel sub-mode selector
                dual_mode_selector = gr.Radio(
                    ["Quick", "Standard", "Detailed"],
                    label="Dual Channel Mode",
                    value="Quick",
                    visible=True,
                    info="Quick: Fast merge in memory | Standard: Sequential with intermediate saves | Detailed: Save all tag sets separately"
                )
                
                with gr.Accordion("⚙️ WD Tagger Settings", open=True):
                    threshold_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=DEFAULT_THRESHOLD, step=0.01, 
                        label="Confidence Threshold", 
                        info="Lower = more tags, higher = more confident tags."
                    )
                    hide_ratings_checkbox = gr.Checkbox(
                        label="Hide Rating Tags (general, sensitive, etc.)", value=False
                    )
                    char_first_checkbox = gr.Checkbox(label="Character Tags First", value=True)
                    no_underscore_checkbox = gr.Checkbox(
                        label="Remove '_' separator from tags", value=False
                    )
                
                with gr.Accordion("⚙️ LLM Tagger Settings", open=False):
                    llm_api_key_textbox = gr.Textbox(
                        label="API Key", type="password", 
                        info="Required for LLM and Dual Channel modes."
                    )
                    llm_api_url_textbox = gr.Textbox(label="API URL", value=DEFAULT_API_URL)
                    llm_model_textbox = gr.Textbox(label="Model", value=DEFAULT_MODEL)
                    llm_prompt_textbox = gr.Textbox(
                        label="LLM Prompt", lines=4, value="", 
                        info="Prompts are loaded from external files. This field is for display only.",
                        interactive=False
                    )
                
                submit_button = gr.Button("🚀 Generate Tags", variant="primary")
            
            with gr.Column(scale=1):
                gr.Markdown("### 3. Get Your Captions")
                status_textbox = gr.Textbox(
                    label="Status", 
                    info="Processing progress will be shown here.", 
                    interactive=False
                )
                download_link = gr.File(label="Download Captions (.zip)", interactive=False)
                
                gr.Markdown(
                    """
                    ### 📖 Mode Explanations
                    
                    **Dual Channel Modes:**
                    - **Quick**: Both taggers run simultaneously, results merged in memory. **Fastest**.
                    - **Standard**: WD tagger → save → LLM tagger → save → intelligent merge. **Sequential & Robust**.
                    - **Detailed**: Save WD tags, LLM tags, and merged tags as separate files. **Most comprehensive output**.
                    
                    **Intelligent Merging**: The LLM analyzes both tag sets and creates an optimized 
                    combination that prioritizes Danbooru consistency while preserving valuable 
                    natural language descriptions.
                    """
                )
        
        # Dynamic visibility for dual mode selector
        def update_dual_mode_visibility(mode):
            return gr.update(visible=(mode == "Dual Channel"))
        
        mode_selector.change(
            fn=update_dual_mode_visibility,
            inputs=[mode_selector],
            outputs=[dual_mode_selector]
        )
        
        # Update prompt display
        def update_prompt_display():
            try:
                default_prompt = load_prompt('default')
                return gr.update(value=default_prompt)
            except FileNotFoundError:
                return gr.update(value="Prompt file not found!")
        
        ui.load(
            fn=update_prompt_display,
            outputs=[llm_prompt_textbox]
        )
        
        submit_button.click(
            fn=process_images_in_parallel, 
            inputs=[
                image_files, mode_selector, dual_mode_selector, threshold_slider, 
                hide_ratings_checkbox, char_first_checkbox, no_underscore_checkbox, 
                llm_api_key_textbox, llm_api_url_textbox, llm_model_textbox
            ], 
            outputs=[status_textbox, download_link]
        )
    
    return ui

if __name__ == "__main__":
    # Initialize with startup banner
    logger.header("UNIVERSAL AI IMAGE TAGGER v2.1", "═", 100)
    logger.info("🚀 Starting application...")
    logger.metric("Session ID", logger.session_id)
    logger.metric("Start time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Initialize WD Tagger
    model_p, tags_p = WDTagger.download_model_and_tags()
    wd_tagger = WDTagger(model_path=model_p, tags_path=tags_p)
    
    logger.success("Application initialization complete!")
    logger.info("Launching Gradio interface...")
    
    # Launch UI
    app_ui = create_ui()
    app_ui.launch()