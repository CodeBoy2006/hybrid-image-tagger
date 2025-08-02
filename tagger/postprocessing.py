import re
from typing import Dict
from pathlib import Path
from . import config
from .data_models import PostProcessSettings
from .utils import logger

def apply_post_processing(tags: str, settings: PostProcessSettings) -> str:
    """Apply comprehensive post-processing to tag strings"""
    if not tags or not tags.strip():
        return ""
    
    logger.debug(f"Applying post-processing: {len(settings._asdict())} settings active")
    
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
    
    if prefix and prefix not in processed_tags:
        processed_tags.insert(0, prefix)
    if suffix and suffix not in processed_tags:
        processed_tags.append(suffix)
    
    result = ', '.join(processed_tags)
    
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