import logging
import torch
import torch.nn.functional as F


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class SimpleBigramModel:
    """
    A tiny "next-token" predictor built from a list of words.

    Plain English summary:
    - We look at pairs of neighboring characters inside words.
    - We learn a simple table of numbers that says: "if the current
      character is X, how likely is the next character Y?" (works for
      any list of words, not just names)
    - With that table, we can estimate how well the model fits the data.
    
    This is NOT a complex AI system. It's just basic counting turned
    into a small math formula so we can train it automatically.

    Key tensors and what they mean:
    - xs: indices for the current character of each bigram example.
    - ys: indices for the next character of each bigram example.
    - W:  a learnable table where row=current char, column=next char.
          Higher numbers in W[row, col] mean the model prefers that
          transition.
    """

    def __init__(self, dataset_path: str = "names.txt", seed: int = 2147483647) -> None:
        # Where we read the training data from (one word per line)
        # Default points to "names.txt" to stay compatible with your current file.
        self.dataset_path = dataset_path
        # Fix the random seed so results are the same each run
        self.rng = torch.Generator().manual_seed(seed)

        # These will be filled after loading and preparing data
        self.chars = None          # list of all characters we use, incl. '.'
        self.stoi = None           # maps character -> index
        self.itos = None           # maps index -> character
        self.xs = None             # tensor of current-character indices
        self.ys = None             # tensor of next-character indices
        self.W = None              # the learnable weight table (our simple model)

    def load_words(self) -> list:
        """Read words from the file. Each line is one word."""
        with open(self.dataset_path, "r") as file:
            words = file.read().splitlines()
        return words

    def build_vocabulary(self, words: list) -> None:
        """Create the list of characters and the two helper look-up tables."""
        # We collect the set of all characters appearing in the words
        unique_chars = sorted(list(set(''.join(words))))
        # We also add '.' which we use to mark the start and end of a name
        self.chars = ['.'] + unique_chars
        self.stoi = {ch: idx for idx, ch in enumerate(self.chars)}
        self.itos = {idx: ch for idx, ch in enumerate(self.chars)}

    def build_training_pairs(self, words: list) -> None:
        """
        Turn words into many small training examples.

        For each word, we add a start marker '.' at the beginning and an
        end marker '.' at the end. Then we look at every adjacent pair
        of characters (current, next) and store them as numbers.
        Example for "anna":
          padded sequence: . a n n a .
          bigrams: (.,a), (a,n), (n,n), (n,a), (a,.)
          xs holds the left char indices, ys holds the right char indices.
        """
        xs, ys = [], []
        for word in words:
            padded = ['.'] + list(word) + ['.']
            for ch_now, ch_next in zip(padded, padded[1:]):
                xs.append(self.stoi[ch_now])
                ys.append(self.stoi[ch_next])

        self.xs = torch.tensor(xs)
        self.ys = torch.tensor(ys)

    def initialize_model(self) -> None:
        """Start with random numbers for our table of next-letter scores."""
        vocab_size = len(self.chars)
        # W is a square table: rows = current letter, columns = next letter
        # requires_grad=True means PyTorch will help us improve W automatically
        self.W = torch.randn((vocab_size, vocab_size), generator=self.rng, requires_grad=True)

    def compute_loss(self) -> torch.Tensor:
        """
        Calculate how wrong the model is on average (lower is better).

        Steps in simple terms:
        - We convert each current-letter index into a row of zeros with a single 1
          (this is called one-hot; it just picks a row from the table).
        - We use the table W to produce a score for each possible next letter.
        - We make the scores positive and normalize them to behave like probabilities.
        - We look up the probability of the real next letter and take the negative log.
        - We average this number across all training pairs. This is our loss.

        Shapes to keep in mind:
        - xenc: [num_pairs, vocab_size]
        - logits/counts/probs: [num_pairs, vocab_size]
        - ys: [num_pairs]
        """
        # 1) Get how many distinct characters we have (including '.')
        vocab_size = len(self.chars)

        # 2) Turn indices into simple selector rows (one-hot): picks a single row of W
        xenc = F.one_hot(self.xs, num_classes=vocab_size).float()

        # 3) Multiply selectors by W to get a score for every possible next character
        logits = xenc @ self.W

        # 4) Make scores positive so they behave like unnormalized probabilities
        counts = logits.exp()

        # 5) Normalize each row so it sums to 1 (now each row is a proper probability list)
        probs = counts / counts.sum(1, keepdim=True)

        # 6) Pick the probability of the real next character for every example,
        #    take a log (so confident wrong answers are punished), and average negative
        loss = -probs[torch.arange(self.xs.size(0)), self.ys].log().mean()
        return loss

    def train(self, steps: int = 30, learning_rate: float = 70.0) -> None:
        """Improve W a little bit for a fixed number of steps."""
        logging.info("Starting training: steps=%d, learning_rate=%.3f", steps, learning_rate)
        for step_index in range(steps):
            loss = self.compute_loss()
            if step_index % max(1, steps // 10) == 0 or step_index == steps - 1:
                logging.info("Step %d/%d - loss: %.6f", step_index + 1, steps, loss.item())

            self.W.grad = None
            loss.backward()

            self.W.data += -learning_rate * self.W.grad
        logging.info("Finished training")

    def fit(self, steps: int = 30, learning_rate: float = 70.0) -> None:
        """Run the full data prep and training pipeline."""
        logging.info("Loading dataset from %s", self.dataset_path)
        words = self.load_words()
        logging.info("Loaded %d words", len(words))

        self.build_vocabulary(words)
        logging.info("Vocabulary size (incl. '.'): %d", len(self.chars))

        self.build_training_pairs(words)
        logging.info("Prepared %d training bigrams", self.xs.size(0))

        self.initialize_model()
        self.train(steps=steps, learning_rate=learning_rate)

    def sample(self, num_examples: int = 10) -> list:
        """
        Make a few example words from the current model.

        How it works in plain English:
        - We start from the special start marker '.'
        - We look up the row of probabilities for the next character
        - We pick the next character at random, following those probabilities
        - We repeat until we pick '.' again, which means "end"
        - We join the picked characters to form one word

        Implementation detail:
        - We exponentiate W and row-normalize it to form P, a matrix of
          next-character probabilities. Then we do ancestral sampling
          from this Markov chain until we hit the end token '.' again.
        """
        if self.W is None or self.chars is None or self.stoi is None or self.itos is None:
            raise RuntimeError("Model is not ready. Run fit() first or set up vocabulary/weights.")

        # Convert the learned scores into probabilities for easy sampling
        counts = self.W.exp()  # make scores positive
        P = counts / counts.sum(1, keepdim=True)  # normalize rows to sum to 1

        start_index = self.stoi['.']
        generated = []
        for _ in range(num_examples):
            ix = start_index
            out_chars = []
            while True:
                # Take the probability row for the current character
                p = P[ix]
                # Randomly pick the next character index according to p
                ix = torch.multinomial(p, num_samples=1, replacement=True, generator=self.rng).item()
                # If we reached the end marker, stop and save the word
                if ix == start_index:
                    break
                # Otherwise collect the character and continue
                out_chars.append(self.itos[ix])
            generated.append(''.join(out_chars))

        return generated


def main() -> None:
    """Train the bigram model and print sample generations."""
    model = SimpleBigramModel(dataset_path="names.txt", seed=2147483647 + 2)
    model.fit(steps=100, learning_rate=50.0)
    samples = model.sample(num_examples=10)
    logging.info("Sampled %d words:", len(samples))
    for word in samples:
        print(word)


if __name__ == "__main__":
    main()