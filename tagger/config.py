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