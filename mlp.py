import logging
import torch
import torch.nn.functional as F  # tensor ops; we'll use one-hot and simple softmax-like normalization


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


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
    4) Log basic shapes so it's easy to verify what's happening.
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
    for word in words[:5]:
            # Start every new word with a context full of '.' (index 0)
            context = [0] * context_window
            # We iterate through all characters of the word plus the final '.' end token
            for ch in word + '.':
                next_index = stoi[ch]
                # Save one training example: current context -> next character index
                inputs.append(context)
                targets.append(next_index)
                logging.debug("context='%s' -> next='%s'",
                              "".join(itos[i] for i in context), itos[next_index])
                # Slide the window forward by dropping the oldest char and appending the new one
                context = context[1:] + [next_index]
            logging.info("Built %d (context, next) pairs so far", len(inputs))
    
    # Turn the python lists into tensors that downstream code can use easily.
    # X has shape [num_examples, context_window] with integer indices
    # Y has shape [num_examples] with the integer index of the next character
    X = torch.tensor(inputs)
    Y = torch.tensor(targets)

    logging.info("Training tensor shapes -> X: %s, Y: %s", tuple(X.shape), tuple(Y.shape))

    # A tiny embedding table that maps character indices -> 2D vectors.
    # In a real model, we'd learn these numbers. Here we just initialize randomly
    # to demonstrate the mechanics of lookup and shape handling.
    embedding_table = torch.randn((27, 2))

    # A quick detour: show how one-hot selects a row from the embedding table.
    # One-hot produces a 27-dim vector with a single 1 at the token index. When
    # we multiply that by the table, it picks the corresponding row (vector).
    f = F.one_hot(torch.tensor(5), num_classes=27).float() @ embedding_table
    logging.debug("Example embedding via one-hot for token index 5 -> vector=%s", f)

    # The idiomatic way: directly index into the embedding table with X.
    # This yields one 2D vector per character in the 3-long context window.
    # Resulting shape is [num_examples, context_window, embedding_dim]
    emb = embedding_table[X]
    logging.info("Embedded context shape: %s (examples, window, dim)", tuple(emb.shape))






if __name__ == "__main__":
    main()