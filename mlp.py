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

TRAIN_STEPS = 20000
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
    
    def standardize_train(x: torch.Tensor, gain: torch.Tensor, bias: torch.Tensor,
                          running_mean: torch.Tensor, running_std: torch.Tensor,
                          momentum: float = 0.1) -> torch.Tensor:
        """Batch standardization for training; updates running stats in-place.

        x: [batch, features]; running_mean/std: [1, features].
        """
        batch_mean = x.mean(0, keepdim=True)
        batch_std = x.std(0, keepdim=True, unbiased=False) 
        # Update running stats in-place for use at eval/generation time
        running_mean.mul_(1.0 - momentum).add_(momentum * batch_mean)
        running_std.mul_(1.0 - momentum).add_(momentum * batch_std)
        return gain * (x - batch_mean) / batch_std + bias

    def standardize_eval(x: torch.Tensor, gain: torch.Tensor, bias: torch.Tensor,
                         running_mean: torch.Tensor, running_std: torch.Tensor) -> torch.Tensor:
        """Standardization for eval/generation using running statistics."""
        return gain * (x - running_mean) / running_std + bias

    def compute_hidden_activations(context_embeddings: torch.Tensor, *, is_training: bool) -> torch.Tensor:
        """Flatten context, apply first linear layer, standardize, then tanh.

        context_embeddings: [batch, CONTEXT_WINDOW, EMBEDDING_DIM] or [CONTEXT_WINDOW, EMBEDDING_DIM] with view used by caller.
        Returns: [batch, HIDDEN_SIZE]
        """
        flattened_context = context_embeddings.view(-1, FLATTENED_CONTEXT_DIM)
        hidden_pre_activation = flattened_context @ weights_input_to_hidden + bias_hidden
        if is_training:
            hidden_pre_activation = standardize_train(hidden_pre_activation, bngain, bnbias, bn_running_mean, bn_running_std)
        else:
            hidden_pre_activation = standardize_eval(hidden_pre_activation, bngain, bnbias, bn_running_mean, bn_running_std)
        return torch.tanh(hidden_pre_activation)

    def compute_logits_from_embeddings(context_embeddings: torch.Tensor, *, is_training: bool) -> torch.Tensor:
        """Convenience: embeddings -> hidden -> logits."""
        hidden_activations = compute_hidden_activations(context_embeddings, is_training=is_training)
        return hidden_activations @ weights_hidden_to_output + bias_output
    
    # First layer: transform from flattened context to hidden representation
    # We use fan_in scaling with a tanh-specific gain (5/3). This mirrors common
    # init schemes (LeCun/He/Xavier variants) to keep the variance of pre-activations
    # approximately stable and avoid tanh saturation at init.
    #   std ≈ gain / sqrt(fan_in), where gain for tanh ≈ 5/3
    weights_input_to_hidden = (
        torch.randn((FLATTENED_CONTEXT_DIM, HIDDEN_SIZE), generator=random_generator)
        * (5/3)
        / (FLATTENED_CONTEXT_DIM ** 0.5)
    )
    bias_hidden = torch.randn(HIDDEN_SIZE, generator=random_generator) * 0.01  # tiny initial offset
    

    
    # Second layer: transform from hidden representation to output logits (vocab_size)
    # One output per character in the vocabulary
    #
    # Small output weights/biases help start the model "uncertain": logits near zero
    # produce near-uniform softmax, which yields healthier, less peaky gradients for cross-entropy.
    # If logits are too large initially, the model can be overconfident and harder to optimize.
    weights_hidden_to_output = torch.randn((HIDDEN_SIZE, vocab_size), generator=random_generator) * 0.01  # small weights → modest logits, better early gradients
    bias_output = torch.randn(vocab_size, generator=random_generator) * 0  # zero bias keeps initial class preferences neutral



    bngain = torch.ones((1,HIDDEN_SIZE))
    bnbias = torch.zeros((1,HIDDEN_SIZE))
    # Running statistics for BN-style standardization (not learned by gradients)
    bn_running_mean = torch.zeros((1, HIDDEN_SIZE))
    bn_running_std = torch.ones((1, HIDDEN_SIZE))
    

    # Collect all trainable tensors. We will learn the embedding table and both layers.
    parameters = [embedding_table, weights_input_to_hidden, bias_hidden, weights_hidden_to_output, bias_output, bngain, bnbias]
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

        # Forward: embeddings -> hidden -> logits
        logits = compute_logits_from_embeddings(context_embeddings, is_training=True)
        
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
    logits = compute_logits_from_embeddings(context_embeddings, is_training=False)
    loss = F.cross_entropy(logits, Y_validation)

    print(loss.item())  # lower is better; use this to compare different runs/settings


    # Generate a few names
    for _ in range(10):
            context = [START_TOKEN_INDEX] * CONTEXT_WINDOW
            generated_indices = []
            while True:
                context_embeddings = embedding_table[torch.tensor(context)]  # [CONTEXT_WINDOW, EMBEDDING_DIM]
                logits = compute_logits_from_embeddings(context_embeddings.view(1, CONTEXT_WINDOW, EMBEDDING_DIM), is_training=False)  # [1, vocab_size]
                probs = F.softmax(logits, dim=1)  # convert logits to probabilities
                ix = torch.multinomial(probs, num_samples=1, generator=random_generator).item()  # sample next char index
                if ix == START_TOKEN_INDEX:
                    break
                generated_indices.append(ix)
                context = context[1:] + [ix]  # slide window to include the new index

            print(''.join(itos[i] for i in generated_indices))





if __name__ == "__main__":
    main()