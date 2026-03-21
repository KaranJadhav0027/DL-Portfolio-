# FLAN-T5 Model: Summarization & Question Answering System

## Overview
This project implements a comprehensive text processing system using Google's FLAN-T5 (Fine-tuned LAnguage Net - Text-to-Text Transfer Transformer) model. The application provides advanced text summarization and context-based question answering capabilities with intelligent model caching for optimal performance.

## Features
- **Text Summarization**: Intelligent text summarization in 4-6 bullet points
- **Question Answering**: Context-based Q&A using local text files
- **Model Caching**: Advanced joblib-based caching for faster subsequent runs
- **Interactive Interface**: User-friendly command-line interface
- **Context Management**: Support for local context files
- **Configurable Parameters**: Adjustable generation parameters for different use cases

## Technical Specifications
- **Model**: Google FLAN-T5-small
- **Framework**: Hugging Face Transformers
- **Caching**: Joblib serialization
- **Tokenization**: AutoTokenizer with parallel processing disabled
- **Generation**: Configurable sampling with top-p and temperature

## Prerequisites
- Python 3.7+
- 8GB+ RAM recommended (model size ~300MB)
- Internet connection for initial model download
- GPU support optional but recommended for better performance

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/SwamitJadhav/Deep_Learning_Models
cd Dl_casestudies/HuggingFace\ LLM/
```

### 2. Install Dependencies
```bash
pip install -r ../../requirements.txt
```

### 3. Verify Installation
```bash
python FLAN_T5.py
```

## Usage

### Basic Usage
```bash
python FLAN_T5.py
```

### Interactive Menu
The application provides an interactive menu with the following options:

1. **Summarize the data**: Text summarization mode
2. **Questions & Answers over local context.txt**: Q&A mode
0. **Exit**: Quit the application

### Summarization Mode
1. Select option 1 from the menu
2. Paste or type the text to summarize
3. Press Enter on an empty line to process
4. View the generated summary

### Question Answering Mode
1. Create a `context.txt` file with your content
2. Select option 2 from the menu
3. Ask questions about the context
4. Receive answers based on the provided context

## Architecture

### Core Components
1. **Model Management**: `load_model_and_tokenizer()`
   - Intelligent model and tokenizer loading
   - Advanced caching with joblib
   - Automatic fallback to Hugging Face download

2. **Text Generation**: `Marvellous_run_flan()`
   - Core FLAN-T5 inference function
   - Configurable generation parameters
   - Optimized tokenization and decoding

3. **Summarization**: `Marvellous_summarize_text()`
   - Specialized text summarization
   - Bullet-point format output
   - Optimized prompt engineering

4. **Context Management**: `Marvellous_load_context()`
   - File-based context loading
   - UTF-8 encoding support
   - Error handling for missing files

5. **Question Answering**: `Marvellous_answer_from_context()`
   - Context-aware Q&A processing
   - Intelligent prompt construction
   - Fallback handling for missing context

6. **User Interface**: `main()`
   - Interactive menu system
   - Input validation and error handling
   - User-friendly experience

### Data Flow
```
User Input → Model Loading → Text Processing → FLAN-T5 Inference → Response Generation → Display
```

## Performance Metrics
- **Model Size**: ~300MB (cached)
- **Memory Usage**: ~1-2GB during inference
- **Processing Speed**: ~2-5 seconds per request
- **Cache Hit Rate**: 100% after first run
- **Context Limit**: ~512 tokens (model dependent)

## Advanced Features

### Caching System
- **First Run**: Downloads and caches model and tokenizer
- **Subsequent Runs**: Loads from cache for instant startup
- **Error Recovery**: Automatic fallback to fresh download

### Generation Parameters
- **Max New Tokens**: 128 (default), 160 (summarization)
- **Sampling**: Top-p sampling with p=0.9
- **Temperature**: 0.7 for balanced creativity/consistency
- **Truncation**: Automatic input truncation

### Context Management
- **File Support**: UTF-8 encoded text files
- **Error Handling**: Graceful handling of missing files
- **Flexible Paths**: Configurable context file location

## Configuration Options

### Model Parameters
```python
MODEL_NAME = "google/flan-t5-small"  # Model selection
max_new_tokens = 128                  # Response length
top_p = 0.9                          # Sampling parameter
temperature = 0.7                    # Creativity level
```

### File Paths
```python
model_path = "flan_model.joblib"      # Model cache
tokenizer_path = "flan_tokenizer.joblib"  # Tokenizer cache
context_path = "context.txt"          # Context file
```

## Troubleshooting

### Common Issues
1. **Model Download Fails**
   - Verify internet connection
   - Check Hugging Face access
   - Clear cache and retry

2. **Memory Issues**
   - Close unnecessary applications
   - Consider using smaller model
   - Check available RAM

3. **Context File Not Found**
   - Create `context.txt` in the same directory
   - Check file permissions
   - Verify file encoding (UTF-8)

4. **Slow Performance**
   - Use GPU acceleration if available
   - Reduce max_new_tokens parameter
   - Check system resources

### Error Messages
- `Context file not found or empty`: Missing or empty context.txt
- `No text received`: Empty input in summarization mode
- `No question received`: Empty input in Q&A mode

## File Structure
```
HuggingFace LLM/
├── FLAN_T5.py                    # Main application file
├── flan_model.joblib            # Cached model file (auto-generated)
├── flan_tokenizer.joblib        # Cached tokenizer file (auto-generated)
├── context.txt                   # Context file for Q&A (user-created)
└── README.md                     # This documentation
```

## Dependencies
- `transformers`: Hugging Face Transformers library
- `torch`: PyTorch backend
- `tokenizers`: Fast tokenization
- `joblib`: Model serialization and caching
- `os`: Environment configuration

## Example Usage

### Summarization Example
```
Input: "Artificial intelligence is transforming industries worldwide. Machine learning algorithms are becoming more sophisticated. Deep learning models can process vast amounts of data. AI applications are expanding into healthcare, finance, and transportation sectors."

Output:
• AI is revolutionizing global industries across multiple sectors
• Machine learning algorithms are advancing in sophistication
• Deep learning enables processing of large-scale datasets
• Applications expanding into healthcare, finance, and transportation
• Technology driving significant transformation worldwide
```

### Question Answering Example
```
Context: "The company reported 25% revenue growth in Q3. New product launches contributed significantly to this growth. Market expansion into Asia-Pacific region was successful."

Question: "What was the revenue growth in Q3?"
Answer: "The company reported 25% revenue growth in Q3."
```

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Author
**Swamit Amit Jadhav**  
Date: 18/09/2025

## Version History
- **v1.0.0** (18/09/2025): Initial release with FLAN-T5 integration and caching

## Support
For issues and questions, please create an issue in the repository or contact the author.

## Future Enhancements
- [ ] Support for multiple context files
- [ ] Batch processing capabilities
- [ ] Custom model fine-tuning
- [ ] Advanced prompt engineering
- [ ] Web interface
- [ ] API endpoint support
- [ ] Multi-language support
- [ ] Performance optimization tools
