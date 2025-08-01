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
from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageOps
from tqdm import tqdm

# Color codes for beautiful output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

# Default API parameters
DEFAULT_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_MAX_TOKENS = 400
DEFAULT_TEMPERATURE = 0.4
DEFAULT_PROMPT_FILE = Path("./prompt.txt")

# Default compression settings
DEFAULT_MAX_DIMS = (2048, 2048)
DEFAULT_QUALITY = 85
MIN_COMPRESSION_SIZE = 2 * 1024 * 1024  # 2MB
TARGET_MAX_FILE_SIZE = 2 * 1024 * 1024  # 2MB

# Default retry parameters
DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_DELAY = 1.0
DEFAULT_BACKOFF_FACTOR = 2.0

def print_banner():
    print(rf"""
{Colors.CYAN}{Colors.BOLD}
    __    __    __  ___    ______                           
   / /   / /   /  |/  /   /_  __/___ _____ _____ ____  _____
  / /   / /   / /|_/ /     / / / __ `/ __ `/ __ `/ _ \/ ___/
 / /___/ /___/ /  / /     / / / /_/ / /_/ / /_/ /  __/ /    
/_____/_____/_/  /_/     /_/  \__/_/\__, /\__, /\___/_/     
                                   /____//____/             {Colors.ENDC}
{Colors.BOLD}AI Image Tagger - Generate detailed tags for your images{Colors.ENDC}
""")

def print_section(title: str):
    print(f"\n{Colors.BLUE}{Colors.BOLD}📋 {title}{Colors.ENDC}")
    print(f"{Colors.BLUE}{'─' * (len(title) + 4)}{Colors.ENDC}")

def print_success(message: str): print(f"{Colors.GREEN}✅ {message}{Colors.ENDC}")
def print_warning(message: str): print(f"{Colors.YELLOW}⚠️  {message}{Colors.ENDC}")
def print_error(message: str): print(f"{Colors.RED}❌ {message}{Colors.ENDC}")
def print_info(message: str): print(f"{Colors.CYAN}ℹ️  {message}{Colors.ENDC}")

def get_input_with_validation(prompt: str, validator=None, default=None, is_path=False, must_exist=False):
    """Enhanced input with validation, defaults, and helpful prompts."""
    while True:
        if default:
            display_prompt = f"{prompt} [{default}]: "
        else:
            display_prompt = f"{prompt}: "
        
        user_input = input(f"{Colors.CYAN}{display_prompt}{Colors.ENDC}").strip()
        
        # Use default if no input provided
        if not user_input and default is not None:
            user_input = str(default)
        
        # Skip validation if empty and no default
        if not user_input and default is None:
            if validator and hasattr(validator, '__name__') and 'required' in validator.__name__:
                print_error("This field is required!")
                continue
            return None
        
        # Path validation
        if is_path:
            path = Path(user_input)
            if must_exist and not path.exists():
                print_error(f"Path does not exist: {path}")
                continue
            return path
        
        # Custom validator
        if validator:
            try:
                if validator(user_input):
                    return user_input
                continue
            except Exception as e:
                print_error(f"Invalid input: {e}")
                continue
        
        return user_input

def validate_api_key_required(value):
    """Validator for required API key."""
    if not value:
        raise ValueError("API key is required")
    return True

def validate_positive_int(value):
    """Validator for positive integers."""
    try:
        num = int(value)
        if num <= 0:
            raise ValueError("Must be a positive integer")
        return True
    except ValueError:
        raise ValueError("Must be a valid positive integer")

def validate_yes_no(value):
    """Validator for yes/no inputs."""
    if value.lower() in ['y', 'yes', 'n', 'no', '']:
        return True
    raise ValueError("Please enter 'y' for yes or 'n' for no")

def load_default_prompt() -> str:
    """Load prompt from default file or return built-in prompt."""
    if DEFAULT_PROMPT_FILE.exists():
        try:
            prompt = DEFAULT_PROMPT_FILE.read_text(encoding='utf-8').strip()
            if prompt:
                print_success(f"Loaded prompt from {DEFAULT_PROMPT_FILE}")
                return prompt
        except Exception as e:
            print_warning(f"Could not read {DEFAULT_PROMPT_FILE}: {e}")
    
    # Built-in fallback prompt
    return "Provide a comma-separated list of descriptive tags for this image. Focus on objects, styles, colors, composition, and artistic elements suitable for AI image generation training."

def compress_image(image_path: Path) -> str:
    """Compress image if >1MB to ~2MB target size, then encode to base64."""
    if image_path.stat().st_size < MIN_COMPRESSION_SIZE:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    try:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img.convert('RGB'))
            
            if img.size[0] > DEFAULT_MAX_DIMS[0] or img.size[1] > DEFAULT_MAX_DIMS[1]:
                img.thumbnail(DEFAULT_MAX_DIMS, Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=DEFAULT_QUALITY, optimize=True)
            
            current_quality = DEFAULT_QUALITY
            while buffer.tell() > TARGET_MAX_FILE_SIZE and current_quality > 30:
                current_quality -= 10
                buffer.seek(0); buffer.truncate(0)
                img.save(buffer, format='JPEG', quality=current_quality, optimize=True)

            compressed_data = buffer.getvalue()
            ratio = (1 - len(compressed_data) / image_path.stat().st_size) * 100
            size_mb = len(compressed_data) / (1024 * 1024)
            print(f"  📦 Compressed {image_path.name}: {ratio:.1f}% reduction → {size_mb:.2f}MB")
            
            return base64.b64encode(compressed_data).decode('utf-8')
            
    except Exception as e:
        print_warning(f"Compression failed for {image_path.name}, using original: {e}")
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

def calculate_backoff(attempt: int) -> float:
    backoff = DEFAULT_INITIAL_DELAY * (DEFAULT_BACKOFF_FACTOR ** attempt)
    return max(0, backoff + random.uniform(-0.1, 0.1) * backoff)

def load_text_from_file(file_path: Optional[Path], file_type: str) -> Optional[str]:
    if not file_path or not file_path.is_file(): return None
    try:
        content = file_path.read_text(encoding='utf-8').strip()
        if not content:
            print_warning(f"{file_type.capitalize()} file is empty: {file_path}")
            return None
        return content
    except Exception as e:
        print_error(f"Error reading {file_type} file {file_path}: {e}")
        return None

def apply_marker_words(tags: str, marker_words: List[str]) -> str:
    if not marker_words: return tags
    marker_prefix = ', '.join(marker_words)
    return f"{marker_prefix}, {tags}" if tags.strip() else marker_prefix

def generate_tags(image_path: Path, api_key: str, api_url: str, model: str, prompt: str) -> Optional[str]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    base64_image = compress_image(image_path)
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        "max_tokens": DEFAULT_MAX_TOKENS, "temperature": DEFAULT_TEMPERATURE
    }

    for attempt in range(DEFAULT_MAX_RETRIES + 1):
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=90)
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content']
            return content.strip().replace('\n', ', ')
        except requests.exceptions.RequestException as e:
            if attempt < DEFAULT_MAX_RETRIES:
                delay = calculate_backoff(attempt)
                print_warning(f"Attempt {attempt+1} failed for {image_path.name}: {e}. Retrying in {delay:.2f}s...")
                time.sleep(delay)
            else:
                print_error(f"Failed to process {image_path.name} after {DEFAULT_MAX_RETRIES+1} attempts.")
        except (KeyError, IndexError, TypeError) as e:
            print_error(f"Invalid API response for {image_path.name}: {e}")
            return None
    return None

def process_image(image_path: Path, output_dir: Path, api_key: str, api_url: str, model: str, prompt: str, skip_existing: bool, marker_words: List[str]) -> bool:
    output_path = output_dir / f"{image_path.stem}.txt"
    if skip_existing and output_path.exists():
        return True
    
    tags = generate_tags(image_path, api_key, api_url, model, prompt)
    if tags:
        output_path.write_text(apply_marker_words(tags, marker_words), encoding='utf-8')
        return True
    return False

def get_image_paths(input_dir: Path) -> List[Path]:
    return sorted([p for ext in IMAGE_EXTENSIONS for p in input_dir.glob(f'*{ext.lower()}')])

def show_configuration_preview(args):
    """Display configuration summary before processing."""
    print_section("Configuration Summary")
    print(f"  📁 Input directory: {Colors.YELLOW}{args.input_dir.resolve()}{Colors.ENDC}")
    print(f"  📁 Output directory: {Colors.YELLOW}{args.output_dir.resolve()}{Colors.ENDC}")
    print(f"  🌐 API URL: {Colors.YELLOW}{args.api_url}{Colors.ENDC}")
    print(f"  🤖 Model: {Colors.YELLOW}{args.model}{Colors.ENDC}")
    print(f"  ⚡ Concurrency: {Colors.YELLOW}{args.concurrency}{Colors.ENDC}")
    print(f"  ⏭️  Skip existing: {Colors.YELLOW}{'Yes' if args.skip_existing else 'No'}{Colors.ENDC}")
    if args.marker_words:
        print(f"  🏷️ Marker words: {Colors.YELLOW}{', '.join(args.marker_words)}{Colors.ENDC}")
    
    # Show image count preview
    image_paths = get_image_paths(args.input_dir)
    print(f"  📸 Images found: {Colors.GREEN}{len(image_paths)}{Colors.ENDC}")
    
    if args.skip_existing:
        existing_count = sum(1 for img in image_paths if (args.output_dir / f"{img.stem}.txt").exists())
        to_process = len(image_paths) - existing_count
        print(f"  📝 Existing tags: {Colors.YELLOW}{existing_count}{Colors.ENDC}")
        print(f"  🆕 To process: {Colors.GREEN}{to_process}{Colors.ENDC}")

def interactive_input():
    print_banner()
    
    # Essential settings
    print_section("Essential Settings")
    input_dir = get_input_with_validation(
        "📁 Input directory with images",
        is_path=True, must_exist=True
    )
    if not input_dir.is_dir():
        print_error(f"Not a directory: {input_dir}")
        sys.exit(1)
    
    # Preview image count immediately
    image_paths = get_image_paths(input_dir)
    if not image_paths:
        print_error("No supported images found in input directory")
        print_info(f"Supported formats: {', '.join(IMAGE_EXTENSIONS)}")
        sys.exit(1)
    
    print_success(f"Found {len(image_paths)} images to process")
    
    output_dir = get_input_with_validation(
        "📁 Output directory for tags",
        default=input_dir,
        is_path=True
    )
    
    api_key = get_input_with_validation(
        "🔑 API key",
        validator=validate_api_key_required
    )
    
    # Advanced settings with smart defaults
    print_section("Advanced Settings")
    print_info("Press Enter to use defaults for quick setup")
    
    api_url = get_input_with_validation(
        "🌐 API URL",
        default=DEFAULT_API_URL
    )
    
    model = get_input_with_validation(
        "🤖 Model name",
        default=DEFAULT_MODEL
    )
    
    # Prompt configuration
    prompt_choice = get_input_with_validation(
        f"📄 Use custom prompt file? (default: {'./prompt.txt' if DEFAULT_PROMPT_FILE.exists() else 'built-in prompt'})",
        validator=validate_yes_no,
        default="n"
    ).lower()
    
    prompt_file = None
    if prompt_choice in ['y', 'yes']:
        prompt_file = get_input_with_validation(
            "📄 Prompt file path",
            is_path=True, must_exist=True
        )
    
    # Marker words
    marker_words_input = get_input_with_validation(
        "🏷️ Marker words to prepend (comma-separated, optional)"
    )
    marker_words = []
    if marker_words_input:
        marker_words = [word.strip() for word in marker_words_input.split(',') if word.strip()]
        print_info(f"Will prepend: {', '.join(marker_words)}")
    
    # Processing options
    print_section("Processing Options")
    
    concurrency = get_input_with_validation(
        "⚡ Concurrent requests",
        validator=validate_positive_int,
        default=4
    )
    
    skip_existing_input = get_input_with_validation(
        "⏭️  Skip images with existing tag files?",
        validator=validate_yes_no,
        default="n"
    ).lower()
    skip_existing = skip_existing_input in ['y', 'yes']
    
    dry_run_input = get_input_with_validation(
        "🧪 Run in preview mode (no API calls)?",
        validator=validate_yes_no,
        default="n"
    ).lower()
    dry_run = dry_run_input in ['y', 'yes']
    
    # Create args object
    args = argparse.Namespace(
        input_dir=input_dir,
        output_dir=output_dir,
        api_key=api_key,
        api_url=api_url,
        model=model,
        prompt_file=prompt_file,
        marker_words=marker_words,
        concurrency=int(concurrency),
        skip_existing=skip_existing,
        dry_run=dry_run
    )
    
    # Show configuration summary
    show_configuration_preview(args)
    
    # Final confirmation
    print_section("Ready to Start")
    
    confirm = get_input_with_validation(
        f"{'🧪 Start preview' if dry_run else '🚀 Start processing'}?",
        validator=validate_yes_no,
        default="y"
    ).lower()
    
    if confirm not in ['y', 'yes']:
        print_info("Operation cancelled by user")
        sys.exit(0)
    
    return args

def main():
    parser = argparse.ArgumentParser(
        description="🏷️ AI Image Tagger - Generate detailed tags for your images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=f"""
Examples:
  {sys.argv[0]} --interactive                    # Interactive mode (recommended)
  {sys.argv[0]} -i ./images -k YOUR_API_KEY      # Quick start
  {sys.argv[0]} -i ./images -k YOUR_API_KEY -s   # Skip existing tags
        """
    )
    
    # Define arguments
    parser.add_argument("--input-dir", "-i", type=Path, help="Directory containing input images")
    parser.add_argument("--api-key", "-k", type=str, help="API key for authentication")
    parser.add_argument("--output-dir", "-o", type=Path, help="Directory to save tag files (default: same as input)")
    parser.add_argument("--interactive", "-I", action="store_true", help="Run in interactive mode (default if no args)")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="API endpoint URL")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Model name to use")
    parser.add_argument("--prompt-file", type=Path, help="Path to custom prompt file")
    parser.add_argument("--marker-words", nargs='*', default=[], help="Marker words to prepend")
    parser.add_argument("--concurrency", "-c", type=int, default=4, help="Number of concurrent API requests")
    parser.add_argument("--skip-existing", "-s", action="store_true", help="Skip images that already have tag files")
    parser.add_argument("--dry-run", "-d", action="store_true", help="Preview actions without making API calls")

    # --- FIX: Prioritize help flag ---
    # Check for help flags before any other logic. If found, print help and exit.
    if '-h' in sys.argv or '--help' in sys.argv:
        parser.print_help()
        sys.exit(0)
    # ---------------------------------

    # Determine if interactive mode should be used
    use_interactive = (
        len(sys.argv) == 1 or
        "--interactive" in sys.argv or "-I" in sys.argv or
        not any(arg in sys.argv for arg in ["--input-dir", "-i"]) or
        not any(arg in sys.argv for arg in ["--api-key", "-k"])
    )
    
    if use_interactive:
        args = interactive_input()
    else:
        args = parser.parse_args()
        
        if not args.input_dir:
            print_error("Input directory is required. Use --interactive for guided setup.")
            sys.exit(1)
        if not args.api_key:
            print_error("API key is required. Use --interactive for guided setup.")
            sys.exit(1)
        
        if not args.output_dir:
            args.output_dir = args.input_dir

    # --- Main Logic (Rest of the function remains the same) ---
    if not args.input_dir.is_dir():
        print_error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prompt_to_use = load_default_prompt()
    if args.prompt_file:
        file_prompt = load_text_from_file(args.prompt_file, "prompt")
        if file_prompt:
            prompt_to_use = file_prompt
            print_success(f"Using custom prompt from: {args.prompt_file.name}")
        else:
            print_warning("Could not load custom prompt file. Using default.")

    image_paths = get_image_paths(args.input_dir)
    if not image_paths:
        print_error("No supported images found in input directory")
        print_info(f"Supported formats: {', '.join(IMAGE_EXTENSIONS)}")
        sys.exit(0)
    
    if not hasattr(args, 'dry_run') or not args.dry_run:
        print_info("Images >2MB will be compressed towards a ~2MB target size.")

    if args.dry_run:
        print_info("🧪 DRY RUN MODE - No API calls will be made.")
        print_section("Preview")
        
        total_to_process = 0
        for img in image_paths:
            output_file = args.output_dir / f"{img.stem}.txt"
            if args.skip_existing and output_file.exists():
                status = f"{Colors.YELLOW}⏭️  skipped (exists){Colors.ENDC}"
            else:
                status = f"{Colors.GREEN}🆕 will process{Colors.ENDC}"
                total_to_process += 1
            
            if len([p for p in image_paths if image_paths.index(p) < 10]) >= len(image_paths[:10]):
                print(f"  {img.name} -> {output_file.name} ({status})")
        
        if len(image_paths) > 10:
            remaining = len(image_paths) - 10
            remaining_to_process = sum(1 for img in image_paths[10:] 
                                     if not (args.skip_existing and (args.output_dir / f"{img.stem}.txt").exists()))
            print(f"  ... and {remaining} more images ({remaining_to_process} to process)")
        
        print_info(f"Total images to process: {total_to_process}/{len(image_paths)}")
        sys.exit(0)

    print_section("Processing Images")
    success_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                process_image, img, args.output_dir, args.api_key, 
                args.api_url, args.model, prompt_to_use, 
                args.skip_existing, args.marker_words
            ): img for img in image_paths
        }
        
        progress_bar = tqdm(
            total=len(futures),
            desc=f"{Colors.GREEN}🔄 Processing{Colors.ENDC}",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {rate_fmt}"
        )
        
        for future in concurrent.futures.as_completed(futures):
            if future.result():
                success_count += 1
            progress_bar.update(1)
        progress_bar.close()

    print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 PROCESSING COMPLETE!{Colors.ENDC}")
    print(f"{Colors.GREEN}✅ Successfully processed: {success_count}/{len(image_paths)} images{Colors.ENDC}")
    
    if (failed := len(image_paths) - success_count) > 0:
        print(f"{Colors.YELLOW}⚠️  Failed or skipped: {failed} images{Colors.ENDC}")
    
    print(f"{Colors.CYAN}📁 Tags saved in: {args.output_dir.resolve()}{Colors.ENDC}")
    
    if success_count > 0:
        print_section("Next Steps")
        print_info("You can now use these tag files with your AI image training pipeline")
        print_info(f"Find your tags in: {args.output_dir}")

if __name__ == "__main__":
    main()
