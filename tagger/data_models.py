from typing import NamedTuple, Optional, Dict

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