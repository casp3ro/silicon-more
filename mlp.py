import torch
import torch.nn.functional as F  # tensor ops; we'll use one-hot and simple softmax-like normalization
import random
import matplotlib.pyplot as plt

# -----------------------------
# Constants
# -----------------------------
PY_RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.9

START_END_TOKEN = '.'
START_TOKEN_INDEX = 0

TORCH_RANDOM_SEED = 2147483647

EMBEDDING_DIM = 10
CONTEXT_WINDOW = 3
HIDDEN_SIZE = 200

TRAIN_STEPS = 15000
BATCH_SIZE = 32
LEARNING_RATE = 0.04

# For the optional one-hot demo
SAMPLE_ONE_HOT_INDEX = 5

# Derived dimensions
FLATTENED_CONTEXT_DIM = CONTEXT_WINDOW * EMBEDDING_DIM

def main() -> None:
    """Tiny, readable demo of how character embeddings work with context.

    What this script shows, step-by-step:
    1) Read a list of names and build a character vocabulary (plus a '.' token
       to mark start/end of a name).
    2) Turn the first few names into many training examples. Each example is:
       - input: the last 3 character indices we've seen (the "context")
       - target: the index of the very next character
    3) Convert those integer indices into a small embedding lookup to get
       a numeric representation (vectors) for each character in the context.
        4) Print a few key shapes so it's easy to verify what's happening.
    """
    # Load the raw dataset (one name per line)
    with open("names.txt", "r") as file:
        words_dataset = file.read().splitlines()

    # Build one vocabulary across the entire dataset so indices are consistent
    characters = sorted(list(set(''.join(words_dataset))))
    characters = [START_END_TOKEN] + characters  # Add start/end token at the beginning
    stoi = {symbol: idx for idx, symbol in enumerate(characters)}  # String -> index
    itos = {idx: symbol for idx, symbol in enumerate(characters)}  # Index -> string

    def build_dataset(words: list, stoi_map: dict, context_window: int = CONTEXT_WINDOW):
        """Turn words into (context, next_char) pairs using a fixed sliding window.

        Args:
            words: List of strings used to produce training samples.
            stoi_map: Mapping from character to integer index.
            context_window: Number of previous characters to condition on.

        Returns:
            X: Tensor of shape [num_samples, context_window] with character indices.
            Y: Tensor of shape [num_samples] with the next-character index.
        """
        # Turn words into (context, next_char) training pairs using a fixed window
        inputs, targets = [], []
        for word in words:
            context = [START_TOKEN_INDEX] * context_window  # start with '.' tokens
            for character in word + '.':
                next_index = stoi_map[character]
                inputs.append(context)
                targets.append(next_index)
                context = context[1:] + [next_index]
        X = torch.tensor(inputs)  # [N, context_window]
        Y = torch.tensor(targets) # [N]
        return X, Y


    random.seed(PY_RANDOM_SEED)  # set Python's RNG so the shuffle order is reproducible
    random.shuffle(words_dataset)  # shuffle so splits are representative

    # Compute split indices for train/val/test partitions
    n1 = int(TRAIN_RATIO * len(words_dataset))
    n2 = int(VAL_RATIO * len(words_dataset))

    training_words = words_dataset[:n1]
    validation_words = words_dataset[n1:n2]
    test_words = words_dataset[n2:]

    X_training, Y_training = build_dataset(training_words, stoi)
    X_validation, Y_validation = build_dataset(validation_words, stoi)
    X_test, Y_test = build_dataset(test_words, stoi)


    
    # Create a random number generator with a fixed seed for reproducible results.
    # This ensures that every time we run the script, we get the same random weights
    # and can compare results across different runs. The seed value (2147483647) is
    # a large prime number that provides good randomness distribution.
    random_generator = torch.Generator().manual_seed(TORCH_RANDOM_SEED)

    # print("Training tensor shapes -> X: %s, Y: %s" % (tuple(X.shape), tuple(Y.shape)))

    # Embedding table maps character indices -> 2D vectors (size = vocabulary size)
    # In a real model, these would be learned; here we initialize randomly.
    vocab_size = len(characters)
    embedding_table = torch.randn((vocab_size, EMBEDDING_DIM), generator=random_generator)

    # A quick detour (optional): one-hot selecting a row from the embedding table.
    # Kept for reference; not printed to avoid noisy output.
    _ = F.one_hot(torch.tensor(SAMPLE_ONE_HOT_INDEX), num_classes=vocab_size).float() @ embedding_table




    # Now we'll build a simple neural network to predict the next character.
    # The network takes the embedded context and learns to output probabilities
    # for each possible next character.
    
    # First layer: transform from flattened context (6D) to hidden representation (100D)
    # We flatten the context because we have 3 characters × 2 embedding dimensions = 6 features
    W1 = torch.randn((FLATTENED_CONTEXT_DIM, HIDDEN_SIZE), generator=random_generator)
    b1 = torch.randn(HIDDEN_SIZE, generator=random_generator)
    

    
    # Second layer: transform from hidden representation (100D) to output logits (vocab_size)
    # One output per character in the vocabulary
    W2 = torch.randn((HIDDEN_SIZE, vocab_size), generator=random_generator)
    b2 = torch.randn(vocab_size, generator=random_generator)
    

    # Collect all trainable tensors. We will learn the embedding table and both layers.
    parameters = [embedding_table, W1, b1, W2, b2]
    # Print total number of trainable parameters for quick sanity check
    print(sum(p.nelement() for p in parameters))

    # Enable gradient tracking for manual optimization below
    for p in parameters:
        p.requires_grad = True  # enable autograd tracking

    # Training loop: forward pass -> backward pass -> parameter update

    for _ in range(TRAIN_STEPS):  # repeat many small learning steps
        # Sample a minibatch of 32 random training examples
        ix = torch.randint(0, X_training.shape[0], (BATCH_SIZE,))

        # The idiomatic way: directly index into the embedding table with X.
        # This yields one 2D vector per character in the 3-long context window.
        # Resulting shape is [num_examples, context_window, embedding_dim]
        context_embeddings = embedding_table[X_training[ix]]


        # Apply the first linear transformation followed by tanh activation
        # Shape: [num_examples, 6] @ [6, 100] + [100] = [num_examples, 100]
        hidden_activations = torch.tanh(context_embeddings.view(-1, FLATTENED_CONTEXT_DIM) @ W1 + b1)


        # Compute the final logits (unnormalized scores for each character)
        # Shapes: [batch, 100] @ [100, vocab_size] + [vocab_size] -> [batch, vocab_size]
        logits = hidden_activations @ W2 + b2
        
        # Cross-entropy loss compares predicted logits vs true next-character targets
        loss = F.cross_entropy(logits, Y_training[ix])

        #BACKWARD PASS
        for p in parameters:
            p.grad = None  # zero gradients from previous step

        loss.backward()

        # Parameter update (simple SGD) with a fixed learning rate
        # Note: we update in-place without an optimizer for clarity
        learning_rate = LEARNING_RATE
        for p in parameters:
            p.data += -learning_rate * p.grad
    

    # Validation: run a forward pass on the validation split to estimate generalization

    context_embeddings = embedding_table[X_validation]
    hidden_activations = torch.tanh(context_embeddings.view(-1, FLATTENED_CONTEXT_DIM) @ W1 + b1)
    logits = hidden_activations @ W2 + b2
    loss = F.cross_entropy(logits, Y_validation)

    print(loss.item())  # lower is better; use this to compare different runs/settings


    # Generate a few names
    for _ in range(10):
            context = [START_TOKEN_INDEX] * CONTEXT_WINDOW
            generated_indices = []
            while True:
                context_embeddings = embedding_table[torch.tensor(context)]  # [CONTEXT_WINDOW, EMBEDDING_DIM]
                hidden_activations = torch.tanh(context_embeddings.view(1, FLATTENED_CONTEXT_DIM) @ W1 + b1)
                logits = hidden_activations @ W2 + b2  # [1, vocab_size]
                probs = F.softmax(logits, dim=1)  # convert logits to probabilities
                ix = torch.multinomial(probs, num_samples=1, generator=random_generator).item()  # sample next char index
                if ix == START_TOKEN_INDEX:
                    break
                generated_indices.append(ix)
                context = context[1:] + [ix]  # slide window to include the new index

            print(''.join(itos[i] for i in generated_indices))





if __name__ == "__main__":
    main()