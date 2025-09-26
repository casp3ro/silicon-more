import random  # for dataset shuffling and sampling seeds
import torch  # main tensor library (like NumPy on GPU/CPU)
import torch.nn.functional as F  # neural network loss functions and utilities

# -----------------------------
# Configuration / hyperparameters
# -----------------------------
PY_RANDOM_SEED = 42  # controls Python's pseudo-random operations (e.g., shuffling)
TRAIN_RATIO = 0.8  # 80% of names for training
VAL_RATIO = 0.9    # next 10% for validation; remainder 10% for test

START_END_TOKEN = '.'  # special token to mark both start and end of a name
START_TOKEN_INDEX = 0  # numeric index reserved for the special token

TORCH_RANDOM_SEED = 2147483647  # fixed seed for reproducible torch RNG

EMBEDDING_DIM = 10      # size of each character's vector representation
CONTEXT_WINDOW = 3      # how many previous characters to condition on
HIDDEN_SIZE = 200       # number of neurons in the hidden layer

TRAIN_STEPS = 20000     # total parameter update steps
BATCH_SIZE = 32         # how many examples per minibatch
LEARNING_RATE = 0.04    # not used directly (we use step LR like makemore)

# For the optional one-hot demo
SAMPLE_ONE_HOT_INDEX = 5

# Derived dimensions
# After embedding each of the CONTEXT_WINDOW characters with EMBEDDING_DIM features,
# we concatenate them, resulting in this flattened feature dimension.
FLATTENED_CONTEXT_DIM = CONTEXT_WINDOW * EMBEDDING_DIM  # features after concat
EPSILON = 1e-5  # small constant to avoid divide-by-zero in normalization

def main() -> None:
    """Train a tiny character-level MLP on names and sample a few outputs.

    High-level flow:
    1) Read names and build a character vocabulary (plus special '.')
    2) Turn names into (context, next_char) training pairs
    3) Define parameters: embeddings C, linear layers W1/W2 (+ b2)
    4) Use simple batch norm before the tanh nonlinearity
    5) Train with SGD and step learning-rate schedule
    6) Validate and sample new names from the model
    """
    # Load the raw dataset (one name per line)
    with open("names.txt", "r") as file:  # one name per line
        words_dataset = file.read().splitlines()

    # Build one vocabulary across the entire dataset so indices are consistent
    characters = sorted(list(set(''.join(words_dataset))))  # unique characters
    characters = [START_END_TOKEN] + characters  # Add start/end token at the beginning
    stoi = {symbol: idx for idx, symbol in enumerate(characters)}  # char -> index
    itos = {idx: symbol for idx, symbol in enumerate(characters)}  # index -> char

    def build_dataset(words: list, stoi_map: dict, context_window: int = CONTEXT_WINDOW):
        """Build (context, next_char) pairs using a fixed sliding window."""
        inputs, targets = [], []  # will hold many (context, next_char_index)
        for word in words:
            # initialize sliding window context with start tokens '.'
            context = [START_TOKEN_INDEX] * context_window
            for character in word + '.':
                next_index = stoi_map[character]
                inputs.append(context)
                targets.append(next_index)
                # slide the window: drop oldest, append new index
                context = context[1:] + [next_index]
        # tensors of integer indices
        X = torch.tensor(inputs)  # shape [num_samples, context_window]
        Y = torch.tensor(targets) # shape [num_samples]
        return X, Y


    random.seed(PY_RANDOM_SEED)  # reproducible Python-level randomness
    random.shuffle(words_dataset)  # shuffle dataset so splits are representative

    # Compute split indices for train/val/test partitions
    n1 = int(TRAIN_RATIO * len(words_dataset))
    n2 = int(VAL_RATIO * len(words_dataset))

    training_words = words_dataset[:n1]
    validation_words = words_dataset[n1:n2]
    test_words = words_dataset[n2:]

    # turn splits into supervised learning pairs
    X_training, Y_training = build_dataset(training_words, stoi)
    X_validation, Y_validation = build_dataset(validation_words, stoi)
    X_test, Y_test = build_dataset(test_words, stoi)


    
    # torch RNG for reproducible parameter init and sampling
    random_generator = torch.Generator().manual_seed(TORCH_RANDOM_SEED)

    # print("Training tensor shapes -> X: %s, Y: %s" % (tuple(X.shape), tuple(Y.shape)))

    vocab_size = len(characters)
    # Embedding table C: maps character indices -> EMBEDDING_DIM vectors
    C = torch.randn((vocab_size, EMBEDDING_DIM), generator=random_generator)

    # simple one-hot lookup demo (not used further, for intuition)
    _ = F.one_hot(torch.tensor(SAMPLE_ONE_HOT_INDEX), num_classes=vocab_size).float() @ C

    class BatchNorm1dSimple:
        """Minimal 1D BatchNorm: normalize per feature using batch stats at train,
        and running stats at eval. Has learnable gain/bias like PyTorch's BatchNorm."""
        def __init__(self, dim: int, eps: float = EPSILON, momentum: float = 0.001) -> None:
            self.eps = eps
            self.momentum = momentum
            self.gain = torch.ones((1, dim))
            self.bias = torch.zeros((1, dim))
            self.running_mean = torch.zeros((1, dim))
            self.running_std = torch.ones((1, dim))

        def __call__(self, x: torch.Tensor, *, is_training: bool) -> torch.Tensor:
            if is_training:
                batch_mean = x.mean(0, keepdim=True)
                batch_std = x.std(0, keepdim=True, unbiased=False).clamp_min(self.eps)
                # update running statistics
                self.running_mean.mul_(1.0 - self.momentum).add_(self.momentum * batch_mean)
                self.running_std.mul_(1.0 - self.momentum).add_(self.momentum * batch_std)
                xhat = (x - batch_mean) / batch_std
            else:
                xhat = (x - self.running_mean) / self.running_std.clamp_min(self.eps)
            return self.gain * xhat + self.bias

    def compute_hidden_activations(context_embeddings: torch.Tensor, *, is_training: bool) -> torch.Tensor:
        """Embedding -> concat -> Linear(W1) -> BatchNorm -> tanh.

        Accepts tensors shaped either [batch, CONTEXT_WINDOW, EMBEDDING_DIM]
        or [CONTEXT_WINDOW, EMBEDDING_DIM] (we'll add batch dim when sampling).
        Returns hidden activations of shape [batch, HIDDEN_SIZE].
        """
        # collapse the context dimension into features for a single Linear layer
        flattened_context = context_embeddings.view(-1, FLATTENED_CONTEXT_DIM)
        hidden_pre_activation = flattened_context @ W1  # Linear layer (no bias)
        hidden_norm = bn(hidden_pre_activation, is_training=is_training)  # BatchNorm
        return torch.tanh(hidden_norm)  # non-linearity

    def compute_logits_from_embeddings(context_embeddings: torch.Tensor, *, is_training: bool) -> torch.Tensor:
        """Convenience wrapper: embeddings -> hidden -> logits over next char."""
        hidden_activations = compute_hidden_activations(context_embeddings, is_training=is_training)
        return hidden_activations @ W2 + b2
    
    # Layer 1
    W1 = (
        torch.randn((FLATTENED_CONTEXT_DIM, HIDDEN_SIZE), generator=random_generator)
        * (5/3)
        / (FLATTENED_CONTEXT_DIM ** 0.5)
    )
    # Note: We omit a bias here because BatchNorm will re-center/shift
    
    # Layer 2
    W2 = torch.randn((HIDDEN_SIZE, vocab_size), generator=random_generator) * 0.01  # small init -> gentler logits
    b2 = torch.randn(vocab_size, generator=random_generator) * 0  # start with neutral class bias



    bn = BatchNorm1dSimple(HIDDEN_SIZE, eps=EPSILON, momentum=0.001)  # BN before tanh
    

    parameters = [C, W1, W2, b2, bn.gain, bn.bias]  # trainable tensors
    print(sum(p.nelement() for p in parameters))  # quick parameter count sanity check

    # Enable gradient tracking for manual optimization below
    for p in parameters:
        p.requires_grad = True  # tell PyTorch to track gradients for SGD

    # -----------------------------
    # Training
    # -----------------------------
    for step in range(TRAIN_STEPS):  # main optimization loop
        # sample a random minibatch of indices
        ix = torch.randint(0, X_training.shape[0], (BATCH_SIZE,))

        # embedding lookup: integer indices -> vectors, shape [B, CONTEXT_WINDOW, EMBEDDING_DIM]
        context_embeddings = C[X_training[ix]]

        # forward pass through the network (training mode uses batch BN stats)
        logits = compute_logits_from_embeddings(context_embeddings, is_training=True)
        
        # cross-entropy compares predicted logits with correct next-char indices
        loss = F.cross_entropy(logits, Y_training[ix])

        # zero out any stale gradients from the previous iteration
        for p in parameters:
            p.grad = None

        # compute fresh gradients via backprop
        loss.backward()

        # SGD parameter update with simple step learning-rate schedule (makemore style)
        learning_rate = 0.1 if step < (TRAIN_STEPS // 2) else 0.01
        for p in parameters:
            p.data += -learning_rate * p.grad
        # periodic logging to monitor training progress
        if (step + 1) % 10000 == 0 or step == 0:
            with torch.no_grad():
                print(f"{step+1:7d}/{TRAIN_STEPS:7d}: {loss.item():.4f}")
    

    # Validation under no-grad with running statistics
    # -----------------------------
    # Validation
    # -----------------------------
    with torch.no_grad():  # no gradients needed for evaluation
        context_embeddings = C[X_validation]
        logits = compute_logits_from_embeddings(context_embeddings, is_training=False)
        val_loss = F.cross_entropy(logits, Y_validation)
    print(val_loss.item())  # lower is better; use to tune hyperparameters


    # -----------------------------
    # Sampling
    # -----------------------------
    def sample_name() -> str:
        """Stochastic decoding loop: sample one character at a time until '.'"""
        with torch.no_grad():
            # start with an all-start-token context of length CONTEXT_WINDOW
            context = [START_TOKEN_INDEX] * CONTEXT_WINDOW
            generated_indices = []
            while True:
                # single example forward pass: shape to [1, CONTEXT_WINDOW, EMBEDDING_DIM]
                context_embeddings = C[torch.tensor(context)]
                logits = compute_logits_from_embeddings(
                    context_embeddings.view(1, CONTEXT_WINDOW, EMBEDDING_DIM),
                    is_training=False,
                )
                # convert logits to probabilities over the vocabulary
                probs = F.softmax(logits, dim=1).squeeze(0)
                # sample the next character index from the distribution
                next_index = torch.multinomial(probs, num_samples=1, generator=random_generator).item()
                # '.' ends the name
                if next_index == START_TOKEN_INDEX:
                    break
                generated_indices.append(next_index)
                # slide the window to include the newest character
                context = context[1:] + [next_index]
        # turn indices back into a string
        return ''.join(itos[i] for i in generated_indices)

    for _ in range(10):
        print(sample_name())





if __name__ == "__main__":
    main()