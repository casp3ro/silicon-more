import torch
import torch.nn.functional as F  # tensor ops; we'll use one-hot and simple softmax-like normalization

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
    with open("names.txt", "r") as file:
        words = file.read().splitlines()

    # Build a vocabulary of characters that appear in the dataset.
    # We also add a special '.' token that will stand for both "start" and "end".
    characters = sorted(list(set(''.join(words))))
    characters = ['.'] + characters  # Add '.' at the beginning for start/end tokens

    stoi = {symbol: idx for idx, symbol in enumerate(characters)}  # String to index mapping
    itos = {idx: symbol for idx, symbol in enumerate(characters)}  # Index to string mapping


    # We'll use a fixed-size rolling window of the previous 3 characters.
    # For example, when reading "emma.", we produce pairs like:
    #   ctx=[., ., .] -> 'e',  ctx=[., ., e] -> 'm',  ... until the final '.'
    context_window = 3
    inputs, targets = [], []
    for word in words:
            # Start every new word with a context full of '.' (index 0)
            context = [0] * context_window
            # We iterate through all characters of the word plus the final '.' end token
            for character in word + '.':
                next_index = stoi[character]
                # Save one training example: current context -> next character index
                inputs.append(context)
                targets.append(next_index)
                # Slide the window forward by dropping the oldest char and appending the new one
                context = context[1:] + [next_index]
    
    # Turn the python lists into tensors that downstream code can use easily.
    # X has shape [num_examples, context_window] with integer indices
    # Y has shape [num_examples] with the integer index of the next character
    X = torch.tensor(inputs)
    Y = torch.tensor(targets)
    
    # Create a random number generator with a fixed seed for reproducible results.
    # This ensures that every time we run the script, we get the same random weights
    # and can compare results across different runs. The seed value (2147483647) is
    # a large prime number that provides good randomness distribution.
    random_generator = torch.Generator().manual_seed(2147483647)

    print("Training tensor shapes -> X: %s, Y: %s" % (tuple(X.shape), tuple(Y.shape)))

    # A tiny embedding table that maps character indices -> 2D vectors.
    # In a real model, we'd learn these numbers. Here we just initialize randomly
    # to demonstrate the mechanics of lookup and shape handling.
    embedding_table = torch.randn((27, 2), generator=random_generator)

    # A quick detour (optional): one-hot selecting a row from the embedding table.
    # Kept for reference; not printed to avoid noisy output.
    _ = F.one_hot(torch.tensor(5), num_classes=27).float() @ embedding_table




    # Now we'll build a simple neural network to predict the next character.
    # The network takes the embedded context and learns to output probabilities
    # for each possible next character.
    
    # First layer: transform from flattened context (6D) to hidden representation (100D)
    # We flatten the context because we have 3 characters × 2 embedding dimensions = 6 features
    W1 = torch.randn((6, 100), generator=random_generator)  # Weight matrix: 6 inputs -> 100 hidden units
    b1 = torch.randn(100, generator=random_generator)       # Bias vector for the hidden layer
    

    
    # Second layer: transform from hidden representation (100D) to output logits (27D)
    # The output has 27 dimensions, one for each character in our vocabulary
    W2 = torch.randn((100, 27), generator=random_generator)  # Weight matrix: 100 hidden -> 27 outputs
    b2 = torch.randn(27, generator=random_generator)         # Bias vector for the output layer
    

    # Collect all trainable tensors. We will learn the embedding table and both layers.
    parameters = [embedding_table, W1, b1, W2, b2]

    # Enable gradient tracking for manual optimization below
    for p in parameters:
        p.requires_grad = True

    # Training loop: forward pass -> backward pass -> parameter update

    for step in range(1000):
        # Sample a minibatch of 32 random training examples
        ix = torch.randint(0, X.shape[0], (32,))

        # The idiomatic way: directly index into the embedding table with X.
        # This yields one 2D vector per character in the 3-long context window.
        # Resulting shape is [num_examples, context_window, embedding_dim]
        context_embeddings = embedding_table[X[ix]]


        # Apply the first linear transformation followed by tanh activation
        # Shape: [num_examples, 6] @ [6, 100] + [100] = [num_examples, 100]
        hidden_activations = torch.tanh(context_embeddings.view(-1, 6) @ W1 + b1)


        # Compute the final logits (unnormalized scores for each character)
        # Shape: [num_examples, 100] @ [100, 27] + [27] = [num_examples, 27]
        logits = hidden_activations @ W2 + b2
        
        # Cross-entropy loss compares predicted logits vs true next-character targets
        loss = F.cross_entropy(logits, Y[ix])
        if step % 100 == 0 or step == 999:
            print("Step %d - loss: %.4f" % (step, loss.item()))

        #BACKWARD PASS
        for p in parameters:
            p.grad = None

        loss.backward()

        # Parameter update (simple SGD) with a fixed learning rate
        # Note: we update in-place without an optimizer for clarity
        learning_rate = 0.1
        for p in parameters:
            p.data += -learning_rate * p.grad
    






if __name__ == "__main__":
    main()