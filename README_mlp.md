## mlp.py — Simple Character-Level Name Model 

This script trains a tiny multilayer perceptron (MLP) to generate names one character at a time. 

### What it does
- Builds a character vocabulary from `names.txt` (one name per line)
- Creates training pairs: previous 3 characters → next character
- Learns:
  - an embedding table `C` to turn character indices into vectors
  - a linear layer `W1` (no bias) + simple BatchNorm + `tanh`
  - a linear output layer `W2` + `b2` to predict the next character
- Trains with SGD and a step learning-rate schedule
- Samples new names by repeatedly predicting the next character until `.`

### File layout (high level)
- Configuration: seeds, hyperparameters (embedding size, context window, hidden size, steps)
- Data prep: read `names.txt`, build `stoi`/`itos`, and create `(X, Y)` pairs
- Parameters: `C`, `W1`, `W2`, `b2`, and a minimal `BatchNorm1dSimple`
- Forward pass helpers: embeddings → hidden → logits
- Training loop: minibatch sampling, forward, loss, backward, SGD update, logging
- Validation: computes validation loss with running BN statistics
- Sampling: generates and prints a few model-created names

### Requirements
- Python 3.9+
- PyTorch

Install (CPU example):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### How to run
1. Ensure `names.txt` is present in the same directory (one name per line).
2. Run the script:
```bash
python mlp.py
```
3. You’ll see training progress prints, a final validation loss, and a few sampled names.

### Key hyperparameters to tweak
- `EMBEDDING_DIM`: vector size for each character (default 10)
- `CONTEXT_WINDOW`: number of previous characters the model sees (default 3)
- `HIDDEN_SIZE`: neurons in the hidden layer (default 200)
- `TRAIN_STEPS`: number of SGD updates (default 20,000)
- Learning rate schedule: 0.1 for the first half of steps, then 0.01

Increasing `HIDDEN_SIZE` or `TRAIN_STEPS` can improve results, at the cost of compute.

### How BatchNorm is used here
- We apply BatchNorm to the pre-activation of the first linear layer (`W1`), then a `tanh` nonlinearity.
- During training, BatchNorm uses batch statistics and also updates running mean/std.
- During validation/sampling, it uses the running statistics for stable behavior.

### Typical outputs
- A printed count of trainable parameters.
- Periodic lines like `   10000/   20000: 2.05` indicating training loss.
- A final validation loss (lower is better).
- 10 sampled names.

### Troubleshooting
- “Import torch could not be resolved”: ensure PyTorch is installed for your environment/interpreter.
- Empty or weird samples: increase `TRAIN_STEPS` or `HIDDEN_SIZE`, or try a different seed.
- Diverging loss early on: confirm `names.txt` exists and uses simple ASCII letters.

### Glossary (basic terms)
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
- **BatchNorm (batch normalization)**: normalizes activations using batch statistics for more stable training.
- **Activation function**: non-linear function; here `tanh`, which squashes values to [-1, 1].
- **Parameters**: the numbers the model learns (e.g., `C`, `W1`, `W2`, `b2`, BN gain/bias).
- **Train/validation/test**: data splits used to train, tune, and finally evaluate the model.
- **Overfitting**: when a model memorizes training data but performs worse on new/unseen data.
- **Seed / generator**: controls randomness so runs are repeatable.
- **Sampling**: using the trained model to generate new names, one character at a time.


