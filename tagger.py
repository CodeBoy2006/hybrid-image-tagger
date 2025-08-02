#!/usr/bin/env python3
from tagger.ui import create_ui
from tagger.utils import logger
from tagger.wd_tagger import WDTagger
from tagger.processing import set_wd_tagger

if __name__ == "__main__":
    logger.header("HYBRID IMAGE TAGGER", "═", 100)
    
    # Download model and tags, and then initialize the WDTagger
    model_path, tags_path = WDTagger.download_model_and_tags()
    wd_tagger_instance = WDTagger(model_path, tags_path)
    
    # Set the wd_tagger instance in the processing module
    set_wd_tagger(wd_tagger_instance)
    
    # Create and launch the UI
    app_ui = create_ui()
    app_ui.launch()