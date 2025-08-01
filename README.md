
# LLM Image Tagger

![Python](https://img.shields.io/badge/Python-3.7\+-00A67E?style=for-the-badge&logo=python&logoColor=white)
[![OpenAI](https://img.shields.io/badge/OpenAI-Compatible-00A67E?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)

A powerful command-line tool that uses AI (VLM) to generate detailed, structured tags for images. Perfect for organizing image datasets, training AI models, or enhancing image metadata.

## Features

- 🏷️ **Detailed Tag Generation**: Creates comprehensive, structured tags for images using advanced AI models
- 🖼️ **Multi-format Support**: Works with JPG, PNG, BMP, WebP, and TIFF images
- 🗜️ **Smart Compression**: Automatically compresses large images (>1MB) to optimize API usage
- ⚡ **Batch Processing**: Process multiple images concurrently with adjustable concurrency
- 🎯 **Customizable Prompts**: Use built-in prompts or provide your own for specialized tagging
- 🏷️ **Marker Words**: Prepend custom tags to all generated tags
- 💾 **Flexible Output**: Save tags alongside images or in a separate directory
- 🔄 **Resume Capability**: Skip images that already have tag files

## Installation

### Prerequisites

- Python 3.7 or higher
- An API key from a compatible AI service (default: OpenAI-compatible API)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Recommended: Interactive Mode

For the best experience, especially for first-time users, run in interactive mode:

```bash
python tagger.py
```

This will guide you through all configuration options with helpful prompts and validation.

### Command Line Interface

For advanced users or automation, you can use command-line arguments:

```bash
python tagger.py --input-dir /path/to/images --api-key YOUR_API_KEY
```

## Core Parameters

| Parameter | Description |
|-----------|-------------|
| `--input-dir` `-i` | Directory containing input images |
| `--api-url` `-u` | The complete base API URL for the requests |
| `--api-key` `-k` | API key for authentication |
| `--output-dir` `-o` | Directory to save tag files (default: same as input) |
| `--concurrency` `-c` | Number of concurrent requests (default: 4) |
| `--skip-existing` `-s` | Skip images that already have tag files |
| `--dry-run` `-d` | Preview actions without making API calls |

## Advanced Options

Additional parameters are available for customization:

```bash
python tagger.py --help
```

## Custom Prompts

You can customize the tagging behavior by creating a `prompt.txt` file in the same directory as the script. The tool will automatically use this file if it exists.

The prompt should instruct the AI on how to analyze and tag images. The default prompt is optimized for generating structured tags suitable for AI image generation models.

## Examples

### Interactive Mode (Recommended)

```bash
python tagger.py
```

### Basic Tagging

```bash
python tagger.py -i ./my_images -k YOUR_API_KEY
```

### Advanced Processing

```bash
python tagger.py \
  -i ./dataset \
  -o ./tags \
  -k YOUR_API_KEY \
  -c 8 \
  -s
```

## Output Format

For each image, the tool generates a `.txt` file with the same name containing comma-separated tags. For example:

```
1girl, anime style, long black hair, blue eyes, school uniform, classroom, daytime, soft lighting, detailed, masterpiece, best quality
```

## Image Compression

Images larger than 2MB are automatically compressed to optimize API usage:

- Images are resized to a maximum of 2048×2048 pixels
- Compressed to JPEG format with adjustable quality
- Target file size is approximately 2MB
- Compression ratio is displayed during processing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.