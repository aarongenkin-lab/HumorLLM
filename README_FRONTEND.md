# HumorLLM

AI-powered humor generation using a custom Seagull transformer architecture with 110M parameters. Features both a modern web interface and command-line tools for generating humorous text completions.

## About

HumorLLM uses modern transformer techniques (RoPE, RMS LayerNorm, SwiGLU) to generate humorous text completions. The model was trained specifically on caption data for humor generation and achieves ~85M parameters with validation perplexity of 13.239.

**Best at:**
- Observational humor ("The thing about grocery shopping is...")
- Absurd situations ("A penguin, a robot, and a toaster walk into...")
- Tech/programming jokes ("My code is so buggy that...")
- Everyday scenarios ("Trying to find a parking spot is like...")

**Style:** Produces quirky, caption-style humor with unexpected observations and surreal commentary.

**Key Characteristics:**
- **Caption comedian**: Excels at describing funny situations like image captions
- **Meta-commentary**: Often explains why things are amusing
- **Visual scenarios**: Works best with descriptive, situational prompts
- **Avoids traditional jokes**: Struggles with classic setup-punchline formats

## Web Frontend

The Flask web interface provides an intuitive way to interact with HumorLLM:

**Features:**
- **Real-time generation** with loading indicators and performance metrics
- **Adjustable controls** for creativity (temperature) and response length
- **Smart text cleaning** automatically removes training artifacts and special tokens
- **Interactive examples** with click-to-try prompts and random generator
- **Modern UI** with responsive design, animations, and glass-morphism effects
- **Keyboard shortcuts** (Ctrl+Enter) and copy-to-clipboard functionality

**Technical Stack:**
- **Backend**: Flask API with improved text post-processing
- **Frontend**: Vanilla JavaScript with modern CSS and animations
- **Performance**: Asynchronous generation with real-time feedback
- **Responsive**: Works on desktop and mobile browsers

## Quick Setup

1. **Install dependencies**
   ```bash
   pip install Flask torch numpy tokenizers transformers
   ```

2. **Run the web interface**
   ```bash
   python app.py
   ```

3. **Open browser**
   Navigate to: http://localhost:5000

## Usage

- Enter a prompt and click "Generate Humor"
- Adjust creativity (temperature) and length controls
- Try example prompts or use the random prompt generator
- Press Ctrl+Enter for quick generation

## Tips for Best Results

**Effective Prompts:**
- Use descriptive scenarios: "The worst thing about X is..."
- Try comparative humor: "Dating apps are like..."
- Set up visual situations: "A robot walks into a coffee shop and..."
- Ask observational questions: "Why do people always..."

**Parameter Settings:**
- **Temperature 0.6-0.7**: More coherent, less random
- **Temperature 0.8-0.9**: Good creativity balance (recommended)
- **Length 20-40 tokens**: Sweet spot for complete thoughts
- **Avoid traditional setups**: "Why did the chicken..." often produces incomplete results

## Command Line

For command line usage:
```bash
python humor_demo.py --prompt "A cat walks into a bar and"
python humor_demo.py --mode interactive
python humor_demo.py --mode benchmark
```

## Performance

- **Speed**: ~14 tokens/second (CPU)
- **Model**: 110M parameters
- **Memory**: ~450MB RAM