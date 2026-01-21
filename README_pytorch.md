## PyTorch for Beginners — Quickstart and Core Methods

This guide is for **beginners who want to understand the pieces used inside `mlp.py`**.  
It explains the most important concepts and APIs in simple terms, with short examples you can try.

You do **not** need to read this to simply run the project, but it is helpful if you are curious about:

- What is a **tensor**?
- What is a **loss**?
- What does **`optimizer.step()`** actually do?

If you want to go deeper later, refer to the official docs:  
[PyTorch documentation](https://docs.pytorch.org/docs/stable/index.html).

---

### 1) What is PyTorch?
PyTorch is a Python library for building and training AI models. You can think of it like "NumPy for AI," with extra powers:
- Tensors: fast arrays that can run on CPU or GPU
- Autograd: automatic differentiation for learning
- nn: building blocks for neural networks
- optim: optimizers for training (SGD, Adam)
- data: tools to load and batch your data

---

### 2) Tensors (torch.Tensor)
Tensors are like powerful arrays.

```python
import torch

# Create tensors
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y = torch.randn(2, 2)              # random values
z = torch.zeros(2, 3)              # all zeros

# Math
s = x + y
p = x @ y.T                        # matrix multiply
m = torch.mean(x)

# Shapes and devices
print(x.shape)                     # torch.Size([2, 2])
print(x.device)                    # cpu (or cuda if on GPU)

# Move to GPU if available
if torch.cuda.is_available():
    x = x.cuda()
```

Key APIs: `torch.tensor`, `torch.zeros/ones/randn`, `.to(device)`, `.cuda()`, `.cpu()`, `.shape`, broadcasting, indexing, slicing.

---

### 3) Autograd (torch.autograd)
Autograd computes gradients automatically — essential for training.

```python
w = torch.tensor(2.0, requires_grad=True)
loss = (w - 5) ** 2               # simple quadratic loss
loss.backward()                   # computes d(loss)/d(w)
print(w.grad)                     # -> tensor(-6.)
```

Key ideas:
- Set `requires_grad=True` for parameters you want to learn
- Compute a scalar loss
- Call `loss.backward()` to populate `.grad`
- Zero grads before the next step: `param.grad = None` or `optimizer.zero_grad()`

---

### 4) Neural Networks (torch.nn)
Use `nn.Module` to define models with layers.

```python
import torch.nn as nn

class TinyNet(nn.Module):
    def __init__(self, in_features: int, hidden: int, out_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )
    def forward(self, x):
        return self.net(x)
```

Common layers/ops:
- Linear: `nn.Linear(in, out)`
- Embedding: `nn.Embedding(num_embeddings, embedding_dim)`
- Convolution: `nn.Conv2d(in_ch, out_ch, kernel_size)`
- Normalization: `nn.BatchNorm1d`, `nn.LayerNorm`
- Activations: `nn.ReLU`, `nn.Tanh`, `nn.GELU`
- Dropout: `nn.Dropout(p)`

---

### 5) Loss Functions (torch.nn.functional / torch.nn)
Loss measures how wrong the model is.

```python
import torch.nn.functional as F

logits = torch.randn(8, 10)      # batch of 8, 10 classes
labels = torch.randint(0, 10, (8,))
loss = F.cross_entropy(logits, labels)
```

Other popular losses:
- `nn.MSELoss()` for regression
- `F.binary_cross_entropy_with_logits` for binary classification

---

### 6) Optimizers (torch.optim)
Optimizers update parameters using gradients.

```python
model = TinyNet(16, 32, 10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for step in range(100):
    logits = model(torch.randn(8, 16))
    labels = torch.randint(0, 10, (8,))
    loss = F.cross_entropy(logits, labels)

    optimizer.zero_grad()   # clear old grads
    loss.backward()         # compute new grads
    optimizer.step()        # update params
```

Popular optimizers:
- `torch.optim.SGD` (with or without momentum)
- `torch.optim.Adam` (adaptive learning rate)

Schedulers (change LR over time):
- `torch.optim.lr_scheduler.StepLR`
- `torch.optim.lr_scheduler.CosineAnnealingLR`

---

### 7) Data Loading (torch.utils.data)
Use `Dataset` and `DataLoader` to handle data and batching.

```python
from torch.utils.data import Dataset, DataLoader

class ToyDataset(Dataset):
    def __init__(self):
        self.x = torch.randn(100, 4)
        self.y = (self.x.sum(dim=1) > 0).long()
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

dataset = ToyDataset()
loader = DataLoader(dataset, batch_size=16, shuffle=True)

for x_batch, y_batch in loader:
    pass  # train here
```

Key APIs: `Dataset`, `DataLoader`, `random_split`, custom collate functions.

---

### 8) Devices (CPU, GPU, MPS)
Move tensors and models to the device you want to use.

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TinyNet(16, 32, 10).to(device)
inputs = torch.randn(8, 16).to(device)
outputs = model(inputs)
```

Also see: `torch.cuda`, `torch.mps` (Apple Silicon), `torch.set_default_device`.

---

### 9) Putting It All Together — Minimal Training Loop

```python
model = TinyNet(16, 32, 10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        logits = model(x_batch)
        loss = F.cross_entropy(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

### 10) Most Useful Cheatsheet
- **Tensors**: `torch.tensor`, `.to(device)`, `.view`, `.reshape`, `.permute`
- **Autograd**: `tensor.requires_grad_()`, `loss.backward()`, `param.grad`
- **Modules**: `nn.Module`, `nn.Sequential`, common layers
- **Losses**: `F.cross_entropy`, `nn.MSELoss`
- **Optim**: `SGD`, `Adam`, `lr_scheduler`
- **Data**: `Dataset`, `DataLoader`, `random_split`
- **Device**: `'cuda'`, `'cpu'`, `'mps'`

For deeper reading and API details, see the official docs: [PyTorch documentation](https://docs.pytorch.org/docs/stable/index.html).

---

### 11) Core Terminology (Beginner-Friendly)
These are the words you’ll see everywhere in AI/ML, explained simply. Think of training as teaching a student with flashcards.

- Dataset: Your whole stack of flashcards (all examples).
- Sample / Example: One flashcard (one input item).
- Feature: What’s on the flashcard (the inputs the model sees). For images: pixels; for text: tokens/characters.
- Label / Target: The answer on the back of the flashcard (what we want the model to predict).
- Batch / Minibatch: A handful of flashcards studied at once (e.g., 32 at a time).
- Epoch: One full pass through all flashcards once.
- Forward pass: The model “guesses” answers for a batch.
- Loss: How wrong the model’s guesses are (lower is better). Cross-entropy is common for classification.
- Backward pass (backprop): The model figures out how to adjust itself to reduce future mistakes.
- Gradient: The direction and amount to change each parameter to improve.
- Optimizer: The rule for updating parameters based on gradients (e.g., SGD, Adam).
- Parameters: The model’s internal numbers that get learned (weights and biases).
- Hyperparameters: The settings you choose, not learned (learning rate, batch size, layers, etc.).
- Learning rate (LR): How big each learning step is. Too big = unstable; too small = slow.
- Scheduler: Changes the learning rate over time (e.g., cosine annealing).
- Overfitting: The model memorizes training flashcards but fails new ones (bad generalization).
- Underfitting: The model is too simple or trained too little; can’t even learn the flashcards.
- Training/Validation/Test split: 
  - Training: used to teach the model.
  - Validation: used to tune settings and check progress.
  - Test: final unbiased check at the end.
- Metric: A number to judge performance (accuracy, loss, F1, etc.).
- Logits: Raw scores before softmax; larger means “more confident.”
- Softmax: Turns logits into probabilities that sum to 1.
- Activation function: Adds non-linearity so models can learn complex patterns (ReLU, Tanh, GELU).
- Normalization: Keeps values in stable ranges (BatchNorm, LayerNorm).
- Regularization: Techniques to reduce overfitting (Dropout, Weight Decay).
- Seed: A fixed number so “random” operations repeat the same way (reproducibility).
- Checkpoint: Saved model state so you can resume or deploy later.
- Device: Where computations run: CPU, GPU (`cuda`), or Apple Silicon (`mps`).

If you want deeper definitions, see the official docs index and tutorials: [PyTorch documentation](https://docs.pytorch.org/docs/stable/index.html).
