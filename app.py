#!/usr/bin/env python3
"""
Flask Web Frontend for HumorLLM
A simple web interface for the HumorLLM humor generation project
"""

from flask import Flask, render_template, request, jsonify
import torch
import json
import sys
import re
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from seagull.model.heads.seagull_lm import SeagullLM
    from seagull.data_processing.bbpe import BBPETokenizer
except ImportError as e:
    print(f"[ERROR] Error importing Seagull modules: {e}")
    sys.exit(1)

app = Flask(__name__)

class HumorLLMAPI:
    """API wrapper for HumorLLM model"""
    
    def __init__(self, model_path="models/final_model.pt", 
                 config_path="config/model_config.json",
                 tokenizer_path="tokenizer/state_dict.json"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.config = None
        self.load_model(model_path, config_path, tokenizer_path)
    
    def load_model(self, model_path, config_path, tokenizer_path):
        """Load model, config, and tokenizer"""
        try:
            # Load config
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            
            # Load tokenizer
            with open(tokenizer_path, 'r') as f:
                state_dict = json.load(f)
            
            self.tokenizer = BBPETokenizer(
                special_tokens=state_dict.get('special_tokens'),
                lowercase=state_dict.get('lowercase', False),
                punct_behavior=state_dict.get('punct_behavior', 'contiguous'),
                name=state_dict.get('name', 'seagull-bbpe')
            )
            
            tokenizer_dir = Path(tokenizer_path).parent
            self.tokenizer.from_file(str(tokenizer_dir))
            
            # Load model
            model_config = self.config['model'].copy()
            model_config['vocab_size'] = self.tokenizer.vocab_size
            
            self.model = SeagullLM(**model_config)
            model_state = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(model_state)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"[OK] HumorLLM loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            raise
    
    def clean_output(self, text):
        """Clean generated text by removing special tokens and artifacts"""
        # Remove special tokens
        text = re.sub(r'<\|[^|]*\|>', '', text)
        
        # Remove repetitive patterns
        words = text.split()
        cleaned_words = []
        prev_phrase = []
        
        for word in words:
            # Check for repetition of 3+ word phrases
            if len(prev_phrase) >= 3 and word == prev_phrase[0]:
                # Skip if we're repeating
                continue
            
            cleaned_words.append(word)
            prev_phrase = (prev_phrase + [word])[-3:]
        
        # Join and clean up
        cleaned = ' '.join(cleaned_words)
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces
        cleaned = re.sub(r'\.+', '.', cleaned)  # Multiple periods
        
        return cleaned.strip()
    
    def generate_humor(self, prompt, max_length=40, temperature=0.7, top_k=50, top_p=0.9):
        """Generate humorous text with improved parameters"""
        if not self.model or not self.tokenizer:
            return {"error": "Model not loaded"}
        
        try:
            self.model.eval()
            
            # Tokenize input
            input_ids = self.tokenizer.encode(prompt)
            generated_ids = input_ids.copy()
            
            with torch.no_grad():
                for _ in range(max_length):
                    # Get model predictions
                    logits, _ = self.model(torch.tensor([generated_ids], device=self.device))
                    logits = logits[0, -1, :] / temperature
                    
                    # Apply top-k filtering
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(logits, top_k)
                        logits = torch.full_like(logits, float('-inf'))
                        logits[top_k_indices] = top_k_logits
                    
                    # Apply top-p filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                        sorted_indices_to_remove[0] = 0
                        
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        logits[indices_to_remove] = float('-inf')
                    
                    # Sample next token
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                    
                    # Stop if EOS token or special tokens
                    if (next_token == self.tokenizer.token2id(self.tokenizer.eos_token) or 
                        next_token in [self.tokenizer.token2id('<|uncanny|>'), 
                                     self.tokenizer.token2id('<|caption|>')]):
                        break
                    
                    generated_ids.append(next_token)
            
            # Decode and clean
            generated_text = self.tokenizer.decode(generated_ids)
            cleaned_text = self.clean_output(generated_text)
            
            return {
                "success": True,
                "result": cleaned_text,
                "prompt": prompt,
                "raw_result": generated_text
            }
            
        except Exception as e:
            return {"error": f"Generation failed: {str(e)}"}

# Initialize the model
humor_api = HumorLLMAPI()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """Generate humor endpoint"""
    data = request.get_json()
    
    if not data or 'prompt' not in data:
        return jsonify({"error": "No prompt provided"}), 400
    
    prompt = data['prompt'].strip()
    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400
    
    # Get optional parameters
    temperature = float(data.get('temperature', 0.7))
    max_length = int(data.get('max_length', 40))
    
    # Clamp values to reasonable ranges
    temperature = max(0.1, min(2.0, temperature))
    max_length = max(10, min(100, max_length))
    
    result = humor_api.generate_humor(
        prompt=prompt,
        max_length=max_length,
        temperature=temperature
    )
    
    return jsonify(result)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": humor_api.model is not None,
        "device": str(humor_api.device)
    })

if __name__ == '__main__':
    print("Starting HumorLLM Web Demo...")
    print("Visit http://localhost:5000 to use the demo")
    app.run(debug=True, host='0.0.0.0', port=5000)