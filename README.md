# HumorLLM

AI-powered humor generation using a custom Seagull transformer architecture with 110M parameters. Features both a modern web interface and command-line tools for generating humorous text completions.

![HumorLLM Demo](https://img.shields.io/badge/Status-Active-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange) ![License](https://img.shields.io/badge/License-MIT-blue)

## Overview

HumorLLM is a specialized transformer model trained specifically for humor generation. Built with a custom "Seagull" architecture, it excels at creating observational humor, absurd situations, and quirky commentary rather than traditional setup-punchline jokes.

### Key Features

🎭 **Specialized for Humor**: Trained on caption data to generate amusing text completions  
🤖 **Modern Architecture**: Custom Seagull transformer with RoPE, RMS LayerNorm, and SwiGLU  
🌐 **Web Interface**: Beautiful Flask frontend with real-time generation  
⚡ **Performance**: 110M parameters, ~14 tokens/second on CPU  
🎛️ **Flexible**: Adjustable creativity and length controls  

## Model Characteristics

**Excels at:**
- **Observational humor**: "The thing about grocery shopping is..."
- **Absurd situations**: "A penguin, a robot, and a toaster walk into..."
- **Tech/programming jokes**: "My code is so buggy that..."
- **Everyday scenarios**: "Trying to find a parking spot is like..."

**Style**: Caption-style humor with unexpected observations and meta-commentary

**Architecture Details:**
- **Model**: Custom Seagull transformer (110M parameters)
- **Techniques**: RoPE positional encoding, RMS LayerNorm, SwiGLU activation
- **Training**: Caption-focused dataset, validation perplexity 13.239
- **Performance**: ~14 tokens/second (CPU), ~450MB memory usage

## Quick Start

### Web Interface (Recommended)

1. **Install dependencies**
   ```bash
   pip install Flask torch numpy tokenizers transformers einops
   ```

2. **Run the web interface**
   ```bash
   python app.py
   ```

3. **Open browser**
   Navigate to: http://localhost:5000

### Command Line Interface

```bash
# Single prompt generation
python humor_demo.py --prompt "A cat walks into a bar and"

# Interactive mode
python humor_demo.py --mode interactive

# Performance benchmark
python humor_demo.py --mode benchmark

# Show model info
python humor_demo.py --mode info
```

## Web Interface Features

The Flask web frontend provides an intuitive experience:

- **Real-time generation** with loading indicators and performance metrics
- **Adjustable controls** for creativity (temperature) and response length
- **Smart text cleaning** that removes training artifacts and special tokens
- **Interactive examples** with click-to-try prompts and random generator
- **Modern UI** with responsive design, animations, and glass-morphism effects
- **Keyboard shortcuts** (Ctrl+Enter) and copy-to-clipboard functionality

## Usage Tips

### Effective Prompts
✅ **Try these formats:**
- Descriptive scenarios: "The worst thing about X is..."
- Comparative humor: "Dating apps are like..."
- Visual situations: "A robot walks into a coffee shop and..."
- Observational questions: "Why do people always..."

❌ **Avoid these formats:**
- Traditional joke setups: "Why did the chicken cross the road?"
- Complex multi-part setups
- Highly abstract concepts

### Parameter Settings
- **Temperature 0.6-0.7**: More coherent, predictable responses
- **Temperature 0.8-0.9**: Good creativity balance (recommended)
- **Temperature 1.0+**: More random, potentially incoherent
- **Length 20-40 tokens**: Sweet spot for complete thoughts

## Project Structure

```
HumorLLM/
├── app.py                      # Flask web application
├── humor_demo.py              # Command-line demo interface
├── templates/                 # Web interface templates
│   └── index.html            # Main web page
├── static/                    # Frontend assets
│   ├── style.css             # Styling and animations
│   └── script.js             # Interactive functionality
├── seagull/                   # Model architecture
│   ├── model/                # Transformer components
│   ├── data_processing/      # Tokenization and data handling
│   ├── nn/                   # Neural network modules
│   ├── trainers/             # Training utilities
│   └── utils/                # Helper functions
├── config/                    # Model configuration
│   └── model_config.json     # Architecture parameters
├── scripts/                   # Training and data scripts
│   ├── train_model.py        # Model training script
│   └── generate_tokenized_dataset.py  # Data preprocessing
├── tokenizer/                 # Tokenizer files
│   ├── tokenizer.json        # Tokenizer configuration
│   └── state_dict.json       # Tokenizer state
├── requirements.txt           # Core dependencies
├── requirements_web.txt       # Web interface dependencies
└── README.md                  # This file
```

## Training and Model Details

### Architecture
- **Base**: Custom Seagull transformer
- **Layers**: 12 transformer layers
- **Embedding Dimension**: 768
- **Attention Heads**: 12
- **FFN Dimension**: 2048
- **Vocabulary Size**: 33,264 tokens
- **Max Sequence Length**: 128 tokens

### Training Configuration
- **Optimizer**: AdamW with weight decay
- **Learning Rate**: Cosine annealing (max 1e-5, min 1e-6)
- **Batch Size**: 16
- **Epochs**: 2
- **Gradient Clipping**: 1.0
- **Mixed Precision**: Enabled
- **Model Compilation**: Enabled for performance

### Performance Metrics
- **Parameters**: 110,528,256 (~110.5M)
- **Validation Perplexity**: 13.239
- **Training Loss**: Converged effectively
- **Generation Speed**: ~14 tokens/second (CPU)

## Development

### Setting Up Development Environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/aarongenkin-lab/HumorLLM.git
   cd HumorLLM
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements_web.txt  # For web interface
   ```

3. **Verify installation**
   ```bash
   python humor_demo.py --mode info
   ```

### Training Your Own Model

The training pipeline is included but requires:
- Caption dataset for humor generation
- Sufficient compute resources (GPU recommended)
- Model weights (not included due to size)

```bash
# Prepare dataset
python scripts/generate_tokenized_dataset.py

# Train model
python scripts/train_model.py
```

## Contributing

We welcome contributions! Areas of interest:
- Improving humor quality and coherence
- Adding new interaction modes
- Optimizing performance
- Expanding training data
- UI/UX enhancements

## Known Limitations

- **Special tokens**: May occasionally include training artifacts
- **Repetition**: Can sometimes repeat phrases or concepts
- **Traditional jokes**: Struggles with classic joke formats
- **Context length**: Limited to 128 tokens max sequence length
- **Model size**: Currently CPU-only optimized

## Future Improvements

- [ ] GPU optimization for faster generation
- [ ] Larger context window support
- [ ] Better special token filtering
- [ ] Multi-modal humor (text + image)
- [ ] Fine-tuning for specific humor styles
- [ ] API endpoints for integration

## License

MIT License - see LICENSE file for details.

## Citation

If you use HumorLLM in your research, please cite:

```bibtex
@software{humorllm2024,
  title={HumorLLM: AI-Powered Humor Generation with Seagull Transformer},
  author={Aaron Genkin},
  year={2024},
  url={https://github.com/aarongenkin-lab/HumorLLM}
}
```

## Acknowledgments

- Built with PyTorch and modern transformer techniques
- Inspired by advances in large language models
- Web interface designed with modern UX principles
- Special thanks to the open-source ML community

---

**Note**: This project includes the model architecture and training code, but pre-trained weights are not included due to file size limitations. You can train your own model using the provided scripts or use the demo with placeholder responses.