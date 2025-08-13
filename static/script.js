// HumorLLM Frontend JavaScript

class HumorLLMApp {
    constructor() {
        this.initializeElements();
        this.bindEvents();
        this.updateCharCounter();
        this.updateSliderValues();
    }

    initializeElements() {
        // Input elements
        this.promptInput = document.getElementById('prompt-input');
        this.temperatureSlider = document.getElementById('temperature');
        this.maxLengthSlider = document.getElementById('max-length');
        this.tempValue = document.getElementById('temp-value');
        this.lengthValue = document.getElementById('length-value');
        this.charCounter = document.getElementById('char-counter');

        // Button elements
        this.generateBtn = document.getElementById('generate-btn');
        this.randomPromptBtn = document.getElementById('random-prompt-btn');
        this.copyBtn = document.getElementById('copy-btn');
        this.toggleDescriptionBtn = document.getElementById('toggle-description');

        // Output elements
        this.outputContainer = document.getElementById('output-container');
        this.generationInfo = document.getElementById('generation-info');
        this.generationTime = document.getElementById('generation-time');
        this.tokenCount = document.getElementById('token-count');

        // Description elements
        this.descriptionDetails = document.getElementById('description-details');

        // Example buttons
        this.exampleItems = document.querySelectorAll('.example-item');

        // Random prompts for the random button
        this.randomPrompts = [
            "The worst thing about being a time traveler is",
            "If cats could talk, they would probably",
            "My WiFi password is so secure that",
            "The real reason dinosaurs went extinct was",
            "Social media would be better if",
            "If aliens visited Earth, they would be confused by",
            "The most useless superpower would be",
            "My phone is so old that",
            "If I had a dollar for every time",
            "The secret to happiness is apparently",
            "Coffee shops are just",
            "Online shopping addiction begins when",
            "If robots took over the world, they would probably",
            "The worst pickup line ever is",
            "My cooking skills are so bad that"
        ];
    }

    bindEvents() {
        // Input events
        this.promptInput.addEventListener('input', () => this.updateCharCounter());
        this.temperatureSlider.addEventListener('input', () => this.updateSliderValues());
        this.maxLengthSlider.addEventListener('input', () => this.updateSliderValues());

        // Button events
        this.generateBtn.addEventListener('click', () => this.generateHumor());
        this.randomPromptBtn.addEventListener('click', () => this.insertRandomPrompt());
        this.copyBtn.addEventListener('click', () => this.copyResult());
        this.toggleDescriptionBtn.addEventListener('click', () => this.toggleDescription());

        // Example prompts
        this.exampleItems.forEach(item => {
            item.addEventListener('click', (e) => {
                const prompt = e.target.getAttribute('data-prompt');
                this.promptInput.value = prompt;
                this.updateCharCounter();
                this.promptInput.focus();
            });
        });

        // Enter key support
        this.promptInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                this.generateHumor();
            }
        });
    }

    updateCharCounter() {
        const length = this.promptInput.value.length;
        this.charCounter.textContent = length;
        
        // Color coding for character count
        if (length > 180) {
            this.charCounter.style.color = '#e74c3c';
        } else if (length > 150) {
            this.charCounter.style.color = '#f39c12';
        } else {
            this.charCounter.style.color = '#666';
        }
    }

    updateSliderValues() {
        this.tempValue.textContent = parseFloat(this.temperatureSlider.value).toFixed(1);
        this.lengthValue.textContent = this.maxLengthSlider.value;
    }

    insertRandomPrompt() {
        const randomIndex = Math.floor(Math.random() * this.randomPrompts.length);
        const randomPrompt = this.randomPrompts[randomIndex];
        
        this.promptInput.value = randomPrompt;
        this.updateCharCounter();
        this.promptInput.focus();
        
        // Add a little animation
        this.promptInput.style.transform = 'scale(1.02)';
        setTimeout(() => {
            this.promptInput.style.transform = 'scale(1)';
        }, 200);
    }

    toggleDescription() {
        const isCollapsed = this.descriptionDetails.classList.contains('collapsed');
        
        if (isCollapsed) {
            this.descriptionDetails.classList.remove('collapsed');
            this.toggleDescriptionBtn.textContent = '−';
            this.toggleDescriptionBtn.title = 'Hide Description';
        } else {
            this.descriptionDetails.classList.add('collapsed');
            this.toggleDescriptionBtn.textContent = '+';
            this.toggleDescriptionBtn.title = 'Show Description';
        }
    }

    async generateHumor() {
        const prompt = this.promptInput.value.trim();
        
        if (!prompt) {
            this.showError('Please enter a prompt first!');
            this.promptInput.focus();
            return;
        }

        // Update UI for loading state
        this.setLoadingState(true);

        const startTime = Date.now();

        try {
            const response = await fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: prompt,
                    temperature: parseFloat(this.temperatureSlider.value),
                    max_length: parseInt(this.maxLengthSlider.value)
                })
            });

            const data = await response.json();
            const endTime = Date.now();
            const generationTimeMs = endTime - startTime;

            if (data.success) {
                this.displayResult(data.result, generationTimeMs, data.prompt);
            } else {
                this.showError(data.error || 'Generation failed');
            }

        } catch (error) {
            console.error('Error:', error);
            this.showError('Network error: Unable to connect to the server');
        } finally {
            this.setLoadingState(false);
        }
    }

    setLoadingState(isLoading) {
        if (isLoading) {
            this.generateBtn.disabled = true;
            this.generateBtn.querySelector('.btn-text').style.display = 'none';
            this.generateBtn.querySelector('.spinner').style.display = 'inline';
            this.outputContainer.classList.add('loading');
        } else {
            this.generateBtn.disabled = false;
            this.generateBtn.querySelector('.btn-text').style.display = 'inline';
            this.generateBtn.querySelector('.spinner').style.display = 'none';
            this.outputContainer.classList.remove('loading');
        }
    }

    displayResult(result, generationTimeMs, originalPrompt) {
        // Clear placeholder content
        this.outputContainer.innerHTML = '';

        // Create result element
        const resultElement = document.createElement('div');
        resultElement.className = 'result-text';
        resultElement.textContent = result;

        this.outputContainer.appendChild(resultElement);

        // Show copy button
        this.copyBtn.style.display = 'block';

        // Show generation info
        this.generationTime.textContent = `Generated in ${(generationTimeMs / 1000).toFixed(2)}s`;
        
        // Estimate token count (rough approximation)
        const tokenCount = Math.round(result.split(' ').length * 1.3);
        this.tokenCount.textContent = `~${tokenCount} tokens`;
        
        this.generationInfo.style.display = 'flex';

        // Store result for copying
        this.lastResult = result;

        // Scroll to result
        this.outputContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    showError(message) {
        this.outputContainer.innerHTML = '';

        const errorElement = document.createElement('div');
        errorElement.className = 'result-text';
        errorElement.style.borderLeftColor = '#e74c3c';
        errorElement.style.background = '#fdf2f2';
        errorElement.innerHTML = `
            <strong>❌ Error:</strong><br>
            ${message}
        `;

        this.outputContainer.appendChild(errorElement);

        // Hide copy button and generation info
        this.copyBtn.style.display = 'none';
        this.generationInfo.style.display = 'none';
    }

    async copyResult() {
        if (!this.lastResult) return;

        try {
            await navigator.clipboard.writeText(this.lastResult);
            
            // Visual feedback
            const originalText = this.copyBtn.textContent;
            this.copyBtn.textContent = '✅ Copied!';
            this.copyBtn.style.background = '#27ae60';
            this.copyBtn.style.color = 'white';
            this.copyBtn.style.borderColor = '#27ae60';

            setTimeout(() => {
                this.copyBtn.textContent = originalText;
                this.copyBtn.style.background = '#f8f9fa';
                this.copyBtn.style.color = '#333';
                this.copyBtn.style.borderColor = '#e1e5e9';
            }, 2000);

        } catch (error) {
            console.error('Failed to copy:', error);
            // Fallback for older browsers
            this.fallbackCopyToClipboard(this.lastResult);
        }
    }

    fallbackCopyToClipboard(text) {
        const textArea = document.createElement('textarea');
        textArea.value = text;
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        try {
            document.execCommand('copy');
            this.copyBtn.textContent = '✅ Copied!';
            setTimeout(() => {
                this.copyBtn.textContent = '📋 Copy';
            }, 2000);
        } catch (error) {
            console.error('Fallback copy failed:', error);
        }
        
        document.body.removeChild(textArea);
    }

    // Health check for the backend
    async checkHealth() {
        try {
            const response = await fetch('/health');
            const data = await response.json();
            
            if (!data.model_loaded) {
                this.showError('Model is not loaded on the server');
            }
            
            return data;
        } catch (error) {
            console.error('Health check failed:', error);
            return null;
        }
    }
}

// Initialize the app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    const app = new HumorLLMApp();
    
    // Perform initial health check
    app.checkHealth().then(health => {
        if (health) {
            console.log('HumorLLM Status:', health);
        }
    });

    // Add some helpful keyboard shortcuts info
    const promptInput = document.getElementById('prompt-input');
    promptInput.title = 'Tip: Press Ctrl+Enter to generate humor quickly!';
});

// Add some fun easter eggs
document.addEventListener('keydown', (e) => {
    // Konami code easter egg (up, up, down, down, left, right, left, right, b, a)
    if (!window.konamiSequence) {
        window.konamiSequence = [];
    }
    
    const konamiCode = [
        'ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown',
        'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight',
        'KeyB', 'KeyA'
    ];
    
    window.konamiSequence.push(e.code);
    
    if (window.konamiSequence.length > konamiCode.length) {
        window.konamiSequence.shift();
    }
    
    if (window.konamiSequence.length === konamiCode.length &&
        window.konamiSequence.every((code, index) => code === konamiCode[index])) {
        
        // Easter egg activated!
        document.body.style.transform = 'rotate(360deg)';
        document.body.style.transition = 'transform 2s ease';
        
        setTimeout(() => {
            document.body.style.transform = '';
            document.body.style.transition = '';
        }, 2000);
        
        // Reset sequence
        window.konamiSequence = [];
        
        // Show a fun message
        const promptInput = document.getElementById('prompt-input');
        const originalValue = promptInput.value;
        promptInput.value = "🎉 Easter egg activated! The AI is now extra funny! 🎭";
        
        setTimeout(() => {
            promptInput.value = originalValue;
        }, 3000);
    }
});