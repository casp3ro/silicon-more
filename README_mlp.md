## `mlp_beginner.py` — Main AI Name Generator (Beginner Friendly)

This file is the **main script you run** in this project.  
It trains a simple AI model that **reads names from `names.txt` and then invents new names**.

You do **not** need to understand machine learning to use it.  
This page explains what it does in **plain language first**, and only then uses technical terms.

If you have not read it yet, see the main project overview in `README` for context.

---

### What `mlp_beginner.py` does (non‑technical view)

- **Reads your names** from `names.txt` (one name per line)
- **Learns patterns** in how letters usually follow each other
- **Practices** by trying to guess the next letter and correcting itself when wrong
- **Generates new names** by “typing” one letter at a time until it decides to stop

You can think of it as **autocomplete that makes up new names** instead of finishing your sentence.

---

### What `mlp_beginner.py` does (technical view)

- Builds a character vocabulary from `names.txt` (one name per line)
- Creates training pairs: previous 3 characters → next character
- Implements a small, clean MLP architecture:
  - `nn.Embedding` layer to turn character indices into vectors
  - Linear layer + `nn.LayerNorm` + `tanh` activation
  - Linear output layer to predict the next character
- Trains with SGD and cosine‑annealing learning rate schedule
- Samples new names by repeatedly predicting the next character until the special `.` symbol

### Architecture overview (how the code is organized)

- **`ModelConfig`**  
  A simple configuration “card” that stores all important settings in one place  
  (how big the model is, how long to train, batch size, learning rate, etc.).

- **`Vocabulary`**  
  Turns characters into numbers and back again (`'a' → 5`, `5 → 'a'`).  
  The model only sees numbers; this class does that translation for you.

- **`CharacterDataset`**  
  Prepares training examples:  
  “previous 3 characters” → “next character”  
  in the standard PyTorch `Dataset` style.

- **`CharacterMLP`**  
  The actual neural network model:
  takes encoded characters as input and outputs a prediction for the next character.

- **Training loop**  
  Handles:
  - Sending data and model to CPU/GPU
  - Doing forward and backward passes
  - Updating weights
  - Printing progress logs to the console

### Requirements

- Python 3.9+
- PyTorch (CPU is enough)

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
python mlp_beginner.py
```
3. You’ll see structured logging with training progress, validation loss, and generated names.

---

### Configuration (changing how powerful the model is)

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

### Key features (why this script is practical)

- **Automatic device detection**: Uses GPU if available, otherwise CPU.
- **Cosine annealing**: Smooth learning‑rate schedule (starts higher, slowly decreases).
- **LayerNorm**: Stable training even with small batch sizes.
- **Structured logging**: Clear progress messages with timestamps and log levels.
- **Error handling**: Friendly error if `names.txt` is missing.
- **Type hints**: Clear function signatures if you read the source code.

---

### What the console output looks like

- Structured logging with timestamps and log levels
- Model parameter count and architecture information
- Periodic training progress, e.g.  
  `Step 1000/20000: Loss=2.05, LR=0.095`
- Final validation loss (lower is better)
- Around 10 generated sample names

---

### Troubleshooting (common issues)

- **“Import torch could not be resolved”**  
  PyTorch is not installed for your Python interpreter.  
  Re‑run the install command or check you are using the right virtual environment.

- **Generated names look empty or very strange**  
  Increase `train_steps` or `hidden_size` in `ModelConfig`, or try a different random seed.

- **Loss explodes or becomes NaN very early**  
  Check that `names.txt` exists, is non‑empty, and uses simple characters (letters, dots, etc.).

- **CUDA out of memory**  
  Reduce `batch_size` or run on CPU (set the device or just run on a machine without CUDA).

---

### Code structure (how to navigate the file)

At a high level, the script follows this pattern:

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

---

### Glossary (jargon translated to plain English)

- **Vocabulary**: the set of unique characters the model can use (letters plus `.`).
- **Embedding**: a learned vector that represents a character; similar characters can end up with similar vectors.
- **Context window**: how many previous characters the model sees to predict the next one (here 3).
- **Forward pass**: compute model outputs from inputs (embeddings → layers → logits → loss).
- **Logits**: raw, unnormalized scores for each class (here, each character) before softmax.
- **Softmax**: turns logits into probabilities that sum to 1.
- **Loss (cross‑entropy)**: measures how wrong the predictions are; lower is better.
- **Backward pass (backprop)**: compute gradients of the loss with respect to parameters.
- **Gradient**: the direction and magnitude to change a parameter to reduce loss.
- **SGD (stochastic gradient descent)**: update rule that nudges parameters in the opposite direction of the gradient.
- **Learning rate (LR)**: how big each learning step is; too big can diverge, too small can be slow.
- **Batch / minibatch**: a subset of training examples processed together in one forward/backward pass.
- **LayerNorm**: normalizes activations across the feature dimension for stable training.
- **Activation function**: non‑linear function; here `tanh`, which squashes values to \([-1, 1]\).
- **Parameters**: the numbers the model learns (embedding weights, linear layer weights/biases).
- **Train/validation/test**: data splits used to train, tune, and finally evaluate the model.
- **Overfitting**: when a model memorizes training data but performs worse on new/unseen data.
- **Seed / generator**: controls randomness so runs are repeatable.
- **Sampling**: using the trained model to generate new names, one character at a time.
- **Device**: CPU or GPU where computations are performed; automatically detected.
- **Cosine annealing**: learning‑rate schedule that smoothly decreases following a cosine curve.


