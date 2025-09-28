## mlp.py — Character-Level Name Model (Industry Best Practices)

This script implements a character-level language model using PyTorch, following patterns used by OpenAI, Anthropic, and other leading AI companies. It trains a multilayer perceptron (MLP) to generate names one character at a time.

### What it does
- Builds a character vocabulary from `names.txt` (one name per line)
- Creates training pairs: previous 3 characters → next character
- Implements a clean MLP architecture:
  - `nn.Embedding` layer to turn character indices into vectors
  - Linear layer + `nn.LayerNorm` + `tanh` activation
  - Linear output layer to predict the next character
- Trains with SGD and cosine annealing learning rate schedule
- Samples new names by repeatedly predicting the next character until `.`

### Architecture Overview
- **`ModelConfig`**: Centralized configuration management using dataclasses
- **`CharacterDataset`**: Proper PyTorch Dataset implementation for data handling
- **`Vocabulary`**: Dedicated vocabulary management class with encode/decode methods
- **`CharacterMLP`**: Clean model class inheriting from `nn.Module`
- **Modular training**: Structured training loop with proper device management
- **Professional logging**: Structured logging with timestamps and levels

### Requirements
- Python 3.9+
- PyTorch

Install (CPU example):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Install (GPU example):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### How to run
1. Ensure `names.txt` is present in the same directory (one name per line).
2. Run the script:
```bash
python mlp.py
```
3. You'll see structured logging with training progress, validation loss, and generated names.

### Configuration
The model uses a `ModelConfig` dataclass for easy hyperparameter tuning:

```python
@dataclass
class ModelConfig:
    # Model architecture
    embedding_dim: int = 10
    context_window: int = 3
    hidden_size: int = 200
    
    # Training hyperparameters
    train_steps: int = 20000
    batch_size: int = 32
    learning_rate: float = 0.1
    min_learning_rate: float = 0.01
    
    # Data splits
    train_ratio: float = 0.8
    val_ratio: float = 0.9
```

### Key Features
- **Automatic device detection**: Uses GPU if available, falls back to CPU
- **Cosine annealing**: Smooth learning rate schedule instead of step function
- **LayerNorm**: More stable than BatchNorm for variable batch sizes
- **Structured logging**: Professional logging with timestamps and levels
- **Error handling**: Proper exception handling for file operations
- **Type safety**: Full type hints throughout the codebase

### How LayerNorm is used here
- We apply LayerNorm to the pre-activation of the first linear layer, then a `tanh` nonlinearity.
- LayerNorm normalizes across the feature dimension, making it stable for any batch size.
- This is more robust than BatchNorm for inference with single samples.

### Typical outputs
- Structured logging with timestamps and log levels
- Model parameter count and architecture information
- Periodic training progress: `Step 1000/20000: Loss=2.05, LR=0.095`
- Final validation loss (lower is better)
- 10 generated sample names

### Troubleshooting
- **"Import torch could not be resolved"**: Ensure PyTorch is installed for your environment/interpreter.
- **Empty or weird samples**: Increase `train_steps` or `hidden_size` in the config, or try a different seed.
- **Diverging loss early on**: Confirm `names.txt` exists and uses simple ASCII letters.
- **CUDA out of memory**: Reduce `batch_size` in the config or use CPU mode.

### Code Structure
The refactored code follows industry best practices:

```python
# Configuration management
config = ModelConfig(hidden_size=512, learning_rate=0.05)

# Clean data handling
vocab = Vocabulary(words)
dataset = CharacterDataset(words, vocab.char_to_idx, config.context_window)

# Modular model
model = CharacterMLP(config)

# Professional training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
```

### Glossary (updated terms)
- **Vocabulary**: the set of unique characters the model can use (letters plus `.`).
- **Embedding**: a learned vector that represents a character; similar characters can end up with similar vectors.
- **Context window**: how many previous characters the model sees to predict the next one (here 3).
- **Forward pass**: compute model outputs from inputs (embeddings → layers → logits → loss).
- **Logits**: raw, unnormalized scores for each class (here, each character) before softmax.
- **Softmax**: turns logits into probabilities that sum to 1.
- **Loss (cross-entropy)**: measures how wrong the predictions are; lower is better.
- **Backward pass (backprop)**: compute gradients of the loss with respect to parameters.
- **Gradient**: the direction and magnitude to change a parameter to reduce loss.
- **SGD (stochastic gradient descent)**: update rule that nudges parameters in the opposite direction of the gradient.
- **Learning rate (LR)**: how big each SGD step is; too big can diverge, too small can be slow.
- **Batch / minibatch**: a subset of training examples processed together in one forward/backward pass.
- **LayerNorm**: normalizes activations across the feature dimension for stable training.
- **Activation function**: non-linear function; here `tanh`, which squashes values to [-1, 1].
- **Parameters**: the numbers the model learns (embedding weights, linear layer weights/biases).
- **Train/validation/test**: data splits used to train, tune, and finally evaluate the model.
- **Overfitting**: when a model memorizes training data but performs worse on new/unseen data.
- **Seed / generator**: controls randomness so runs are repeatable.
- **Sampling**: using the trained model to generate new names, one character at a time.
- **Device**: CPU or GPU where computations are performed; automatically detected.
- **Cosine annealing**: learning rate schedule that smoothly decreases following a cosine curve.


