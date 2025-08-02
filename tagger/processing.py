import concurrent.futures
import tempfile
import time
import traceback
import zipfile
from pathlib import Path
import gradio as gr
from PIL import Image, ImageOps
from tqdm import tqdm
from . import config
from .data_models import TaggingResult, PostProcessSettings
from .llm_engine import generate_llm_tags, merge_tags_intelligent
from .postprocessing import apply_post_processing, parse_custom_replacements
from .llm_engine import load_prompt
from .utils import logger
from .wd_tagger import WDTagger

wd_tagger: WDTagger

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

def process_images_in_parallel(
    files, mode, dual_mode, max_workers, wd_threshold, hide_ratings, char_first, no_underscore, 
    llm_api_key, llm_api_url, llm_model,
    # Post-processing parameters
    pp_replace_underscores, pp_escape_brackets, pp_normalize_spaces, pp_remove_duplicates, 
    pp_sort_alphabetically, pp_trigger_prefix, pp_trigger_suffix, pp_custom_replacements,
    pp_max_tags, pp_min_tag_length,
    progress=gr.Progress(track_tqdm=True)
):
    global wd_tagger
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
                    print(f"\n{config.Colors.BG_RED}{config.Colors.BOLD}{config.Colors.WHITE}--- TRACEBACK for {filename} ---{config.Colors.RESET}\n{config.Colors.BRIGHT_RED}{tb_str}{config.Colors.RESET}\n")
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

def set_wd_tagger(tagger: WDTagger):
    global wd_tagger
    wd_tagger = tagger