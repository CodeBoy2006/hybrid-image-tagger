from pathlib import Path
from typing import Tuple
import numpy as np
import onnxruntime as ort
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image
from . import config
from .utils import logger

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
        
        repo_id = config.WD_MODEL_REPO
        filenames = [config.WD_MODEL_FILENAME, config.WD_TAGS_FILENAME]
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
        new_height = int(ratio * config.WD_MODEL_IMG_SIZE)
        processed_image = image.resize((config.WD_MODEL_IMG_SIZE, new_height), Image.Resampling.BICUBIC)
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