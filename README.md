# Hybrid Image Tagger

![Python](https://img.shields.io/badge/Python-3.10+-00A67E?style=for-the-badge&logo=python&logoColor=white)[![OpenAI](https://img.shields.io/badge/OpenAI-Compatible-00A67E?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)![ONNX](https://img.shields.io/badge/ONNX-RUNTIME-00A67E?style=for-the-badge&logo=ONNX&logoColor=white)![Gradio](https://img.shields.io/badge/Gradio-UI-00A67E?style=for-the-badge&logo=gradio&logoColor=white)

A powerful and user-friendly tool that uses a hybrid approach, combining the strengths of the WD 1.4 Tagger and a Vision Language Model (VLM), to generate detailed and accurate tags for images. The tool is wrapped in a Gradio UI for ease of use.

## Features

-   **Hybrid Tagging**: Utilizes both WD Tagger and a VLM for comprehensive and high-quality tag generation.
-   **Dual Channel Processing**: Choose between different strategies for combining the taggers, including parallel and sequential processing.
-   **Advanced Post-Processing**: A rich set of options to refine tags, including custom replacements, trigger words, and more.
-   **User-Friendly UI**: A Gradio interface for easy configuration and use.
-   **Batch Processing**: Process multiple images concurrently with adjustable concurrency.
-   **Smart Compression**: Automatically compresses large images to optimize API usage.

## Installation

### Prerequisites

-   Python 3.10 or higher
-   An API key from a compatible AI service (e.g., OpenAI) for the VLM tagger.

### Recommended Setup

It is highly recommended to use a Python virtual environment (`venv`) to avoid conflicts with other projects and system-wide packages.

1.  **Create a Virtual Environment**

    From your project's root directory, run:
    ```bash
    python -m venv venv
    ```

2.  **Activate the Virtual Environment**

    The activation command depends on your operating system:

    -   **On Windows (Command Prompt or PowerShell):**
        ```bash
        .\venv\Scripts\activate
        ```
    -   **On macOS and Linux:**
        ```bash
        source venv/bin/activate
        ```
    Your terminal prompt should now be prefixed with `(venv)`.

3.  **Install Dependencies**

    With the virtual environment active, install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To launch the application, run the following command:

```bash
python tagger.py
```

This will start the Gradio web UI, which you can access in your browser.

## The Interface

The Gradio interface is divided into three main sections:

1.  **Upload & Configure**: Upload your images and configure the tagging and post-processing settings.
2.  **Processing Status**: Monitor the progress of the tagging process.
3.  **Download Results**: Download the generated tags as a zip file.

### Tagging Modes

-   **WD Tagger Only**: Uses only the WD 1.4 Tagger.
-   **LLM Only**: Uses only the VLM tagger.
-   **Dual Channel**: Uses both taggers. You can choose between three strategies:
    -   **Quick**: Runs both taggers in parallel for each image.
    -   **Standard**: Runs the taggers sequentially for each image.
    -   **Detailed**: Runs both taggers in parallel and saves all intermediate files.

### Post-Processing

A wide range of post-processing options are available to clean and refine the generated tags:

-   **Text Formatting**: Replace underscores, escape brackets, normalize spaces, remove duplicates, and sort alphabetically.
-   **Trigger Words**: Add custom prefixes and suffixes to your tags.
-   **Advanced**: Set custom text replacements, and limits for the maximum number of tags and minimum tag length.

## Output Format

For each image, the tool generates a `.txt` file with the same name containing comma-separated tags.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.