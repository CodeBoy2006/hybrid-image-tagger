import gradio as gr
from . import config
from .processing import process_images_in_parallel

def create_ui():
    with gr.Blocks(title="Hybrid Image Tagger", theme=gr.themes.Soft()) as ui:
        gr.Markdown('''
<div style="background: linear-gradient(135deg, #22d3ee 0%, #6366f1 100%); padding: 24px; border-radius: 16px; margin-bottom: 20px; color: white; box-shadow: 0 8px 32px rgba(99, 102, 241, 0.12);">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 20px;">
        <div>
            <h1 style="margin: 0; font-size: 2.4em; font-weight: 600; text-shadow: none; letter-spacing: -0.02em;">
                🏷️ Hybrid Image Tagger
            </h1>
            <p style="margin: 12px 0 0 0; font-size: 1.1em; opacity: 0.92; font-weight: 400; line-height: 1.5; max-width: 600px;">
                A versatile tool for superior image tagging, utilizing WD tagger and VLM with advanced post-processing.
            </p>
        </div>
        <div>
            <a href="https://github.com/CodeBoy2006/hybrid-image-tagger" target="_blank" rel="noopener noreferrer" 
               style="display: inline-block; transition: all 0.3s ease; text-decoration: none; transform: translateY(0);"
               onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 8px 25px rgba(0,0,0,0.15)'" 
               onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.1)'">
                <img src="https://img.shields.io/badge/View_on_GitHub-181717?style=for-the-badge&logo=github&logoColor=white" 
                     alt="View on GitHub" style="border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
            </a>
        </div>
    </div>
</div>
''')
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
                        threshold_slider = gr.Slider(minimum=0.1, maximum=1.0, value=config.DEFAULT_THRESHOLD, step=0.01, label="Confidence Threshold")
                        hide_ratings_checkbox = gr.Checkbox(label="Hide Rating Tags", value=False)
                        char_first_checkbox = gr.Checkbox(label="Character Tags First", value=True)
                        no_underscore_checkbox = gr.Checkbox(label="Remove '_' from tags", value=False)
                    with gr.Tab("LLM Tagger"):
                        llm_api_key_textbox = gr.Textbox(label="API Key", type="password", info="Required for LLM and Dual Channel modes.")
                        llm_api_url_textbox = gr.Textbox(label="API URL", value=config.DEFAULT_API_URL)
                        llm_model_textbox = gr.Textbox(label="Model", value=config.DEFAULT_MODEL)

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