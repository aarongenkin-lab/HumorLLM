#!/usr/bin/env python3
"""
HumorLLM: A Transformer-Based Language Model for Humor Generation

This script demonstrates the HumorLLM project - a custom transformer architecture 
called "Seagull" for generating humorous captions and text. The model uses modern 
techniques including RoPE positional encoding, RMS normalization, and SwiGLU activation.

Key Features:
- Custom Transformer Architecture: Seagull transformer with 12 layers, 768 embedding dimensions
- Modern Techniques: RoPE, RMS LayerNorm, SwiGLU FFN, Gradient Clipping
- Optimized Training: Mixed precision training, model compilation, cosine LR scheduling
- Humor-Focused: Trained specifically on caption data for humor generation

Best Performance: validation loss 2.559, perplexity 13.239 (~85M parameters)
"""

import torch
import torch.nn as nn
import json
import sys
import time
import argparse
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from seagull.model.heads.seagull_lm import SeagullLM
    from seagull.data_processing.bbpe import BBPETokenizer
except ImportError as e:
    print(f"[ERROR] Error importing Seagull modules: {e}")
    print("Please ensure you're running from the project root directory.")
    sys.exit(1)


class HumorLLMDemo:
    """Main demo class for HumorLLM project"""
    
    def __init__(self, model_path: str = "models/final_model.pt", 
                 config_path: str = "config/model_config.json",
                 tokenizer_path: str = "tokenizer/state_dict.json"):
        """
        Initialize the HumorLLM demo
        
        Args:
            model_path: Path to trained model weights
            config_path: Path to model configuration
            tokenizer_path: Path to tokenizer state dict
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.config = None
        
        print("HumorLLM Demo - Transformer-Based Humor Generation")
        print("=" * 60)
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
        print()
        
        self._load_config(config_path)
        self._load_tokenizer(tokenizer_path)
        self._load_model(model_path)
    
    def _load_config(self, config_path: str):
        """Load model configuration"""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            print("[OK] Configuration loaded successfully")
        except FileNotFoundError:
            print(f"[ERROR] Config file not found: {config_path}")
            sys.exit(1)
    
    def _load_tokenizer(self, tokenizer_path: str):
        """Load and initialize tokenizer"""
        try:
            # Load state dict first
            with open(tokenizer_path, 'r') as f:
                state_dict = json.load(f)
            
            # Initialize tokenizer with state dict
            self.tokenizer = BBPETokenizer(
                special_tokens=state_dict.get('special_tokens'),
                lowercase=state_dict.get('lowercase', False),
                punct_behavior=state_dict.get('punct_behavior', 'contiguous'),
                name=state_dict.get('name', 'seagull-bbpe')
            )
            
            # Load the actual trained tokenizer
            tokenizer_dir = Path(tokenizer_path).parent
            self.tokenizer.from_file(str(tokenizer_dir))
            
            print(f"[OK] Tokenizer loaded (vocab size: {self.tokenizer.vocab_size})")
        except FileNotFoundError:
            print(f"[ERROR] Tokenizer file not found: {tokenizer_path}")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Error loading tokenizer: {e}")
            sys.exit(1)
    
    def _load_model(self, model_path: str):
        """Load and initialize the trained model"""
        try:
            # Initialize model with config
            model_config = self.config['model'].copy()
            model_config['vocab_size'] = self.tokenizer.vocab_size
            
            self.model = SeagullLM(**model_config)
            
            # Load trained weights
            model_state = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(model_state)
            self.model.to(self.device)
            self.model.eval()
            
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            print(f"[OK] Model loaded successfully")
            print(f"[INFO] Total Parameters: {total_params:,} (~{total_params/1e6:.1f}M)")
            print(f"[INFO] Trainable Parameters: {trainable_params:,}")
            
        except FileNotFoundError:
            print(f"[ERROR] Model file not found: {model_path}")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Error loading model: {e}")
            sys.exit(1)
    
    def generate_humor(self, prompt: str, max_length: int = 50, 
                      temperature: float = 0.8, top_k: int = 50, 
                      top_p: float = 0.9) -> str:
        """
        Generate humorous text continuation from a prompt
        
        Args:
            prompt: Input text prompt
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_k: Top-k sampling
            top_p: Nucleus sampling
            
        Returns:
            Generated humorous text
        """
        self.model.eval()
        
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=self.device)
        
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
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                # Stop if EOS token
                if next_token == self.tokenizer.token2id(self.tokenizer.eos_token):
                    break
                    
                generated_ids.append(next_token)
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated_ids)
        return generated_text
    
    def show_model_info(self):
        """Display detailed model information"""
        print("\n[ARCHITECTURE] Model Architecture Information")
        print("=" * 50)
        
        model_config = self.config['model']
        print("[CONFIG] Configuration:")
        for key, value in model_config.items():
            print(f"  {key}: {value}")
        
        print(f"\n[TOKENIZER] Tokenizer:")
        print(f"  Vocabulary Size: {self.tokenizer.vocab_size}")
        print(f"  EOS Token: {self.tokenizer.eos_token} (ID: {self.tokenizer.token2id(self.tokenizer.eos_token)})")
        print(f"  PAD Token: {self.tokenizer.pad_token} (ID: {self.tokenizer.token2id(self.tokenizer.pad_token)})")
        print(f"  UNK Token: {self.tokenizer.unk_token} (ID: {self.tokenizer.token2id(self.tokenizer.unk_token)})")
        
        print(f"\n[PROMPT] Training Configuration:")
        train_config = self.config['train_and_eval']
        for key, value in train_config.items():
            print(f"  {key}: {value}")
    
    def run_demo_prompts(self):
        """Run demonstration with predefined prompts"""
        demo_prompts = [
            "A cat walks into a bar and",
            "Why did the programmer quit his job?",
            "The funniest thing about artificial intelligence is",
            "My computer is so slow that",
            "A robot, a human, and a cat are in an elevator when"
        ]
        
        print("\n[DEMO] Demo: Generating Humorous Completions")
        print("=" * 60)
        
        for i, prompt in enumerate(demo_prompts, 1):
            print(f"\n[PROMPT] Prompt {i}: {prompt}")
            print("-" * 40)
            
            # Generate with different temperatures
            for temp_name, temp_val in [("Conservative", 0.6), ("Balanced", 0.8), ("Creative", 1.0)]:
                try:
                    completion = self.generate_humor(prompt, max_length=30, temperature=temp_val)
                    print(f"  {temp_name} (T={temp_val:.1f}): {completion}")
                except Exception as e:
                    print(f"  [ERROR] Error: {e}")
            print()
    
    def run_benchmark(self, num_runs: int = 5):
        """Run performance benchmark"""
        print(f"\n[BENCHMARK] Performance Benchmark ({num_runs} runs)")
        print("=" * 40)
        
        test_prompt = "The funniest thing about"
        total_time = 0
        total_tokens = 0
        
        for i in range(num_runs):
            start_time = time.time()
            result = self.generate_humor(test_prompt, max_length=20, temperature=0.8)
            end_time = time.time()
            
            generation_time = end_time - start_time
            tokens_generated = len(self.tokenizer.encode(result)) - len(self.tokenizer.encode(test_prompt))
            
            total_time += generation_time
            total_tokens += tokens_generated
            
            print(f"  Run {i+1}: {generation_time:.3f}s, {tokens_generated} tokens")
        
        avg_time = total_time / num_runs
        avg_tokens = total_tokens / num_runs
        tokens_per_second = avg_tokens / avg_time if avg_time > 0 else 0
        
        print(f"\n[STATS] Results:")
        print(f"  Average time: {avg_time:.3f}s")
        print(f"  Average tokens: {avg_tokens:.1f}")
        print(f"  Speed: {tokens_per_second:.1f} tokens/second")
    
    def interactive_mode(self):
        """Run interactive humor generation"""
        print("\n[INTERACTIVE] Interactive Humor Generator")
        print("Enter your prompts below (type 'quit' to exit)")
        print("=" * 50)
        
        while True:
            try:
                user_prompt = input("\n[PROMPT] Your prompt: ").strip()
                
                if user_prompt.lower() in ['quit', 'exit', 'q']:
                    print("[GOODBYE] Thanks for using HumorLLM!")
                    break
                
                if user_prompt:
                    print("[HUMOR] Generating humor...")
                    start_time = time.time()
                    completion = self.generate_humor(user_prompt, max_length=40, temperature=0.8)
                    end_time = time.time()
                    
                    print(f"[RESULT] Result: {completion}")
                    print(f"[TIME]  Generated in {end_time - start_time:.2f}s")
                    print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n[GOODBYE] Goodbye!")
                break
            except Exception as e:
                print(f"[ERROR] Error: {e}")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="HumorLLM Demo - Transformer-based Humor Generation")
    parser.add_argument('--model', default='models/final_model.pt', 
                       help='Path to model weights')
    parser.add_argument('--config', default='config/model_config.json',
                       help='Path to model config')
    parser.add_argument('--tokenizer', default='tokenizer/state_dict.json',
                       help='Path to tokenizer state dict')
    parser.add_argument('--mode', choices=['demo', 'interactive', 'benchmark', 'info'], 
                       default='demo', help='Demo mode to run')
    parser.add_argument('--prompt', type=str, help='Single prompt to generate from')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature (default: 0.8)')
    parser.add_argument('--max-length', type=int, default=50,
                       help='Maximum tokens to generate (default: 50)')
    
    args = parser.parse_args()
    
    # Initialize demo
    demo = HumorLLMDemo(args.model, args.config, args.tokenizer)
    
    try:
        if args.prompt:
            # Single prompt generation
            print(f"\n[PROMPT] Prompt: {args.prompt}")
            result = demo.generate_humor(args.prompt, args.max_length, args.temperature)
            print(f"[RESULT] Result: {result}")
            
        elif args.mode == 'info':
            demo.show_model_info()
            
        elif args.mode == 'benchmark':
            demo.run_benchmark()
            
        elif args.mode == 'interactive':
            demo.interactive_mode()
            
        else:  # demo mode
            demo.show_model_info()
            demo.run_demo_prompts()
            demo.run_benchmark()
            
            # Ask if user wants interactive mode
            response = input("\n[INTERACTIVE] Would you like to try interactive mode? (y/n): ")
            if response.lower().startswith('y'):
                demo.interactive_mode()
    
    except Exception as e:
        print(f"[ERROR] Demo error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()