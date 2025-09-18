import torch
import torch.nn.functional as F


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
        for _ in range(steps):
            # a) Measure how wrong we are right now
            loss = self.compute_loss()
            # b) Print current loss so you can see it decrease over time
            print(loss.item())

            # c) Ask PyTorch to compute how to change W to reduce the loss
            self.W.grad = None
            loss.backward()

            # d) Move W a small step in the better direction
            self.W.data += -learning_rate * self.W.grad

    def fit(self, steps: int = 30, learning_rate: float = 70.0) -> None:
        """Convenience method: run the whole setup and training."""
        # 1) Read words (one per line) from the dataset file
        words = self.load_words()
        # 2) Build the list of characters and index lookups
        self.build_vocabulary(words)
        # 3) Convert words into many (current_char, next_char) pairs
        self.build_training_pairs(words)
        # 4) Start with a random table of scores
        self.initialize_model()
        # 5) Improve the table by a few training steps
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
    # Create the model and train it. Defaults are kept simple on purpose.
    # You can pass a different file via dataset_path (one word per line).
    model = SimpleBigramModel(dataset_path="names.txt", seed=2147483647 +2)
    model.fit(steps=100, learning_rate=50.0)
    # After training, print a few made-up examples
    for w in model.sample(num_examples=10):
        print(w)


if __name__ == "__main__":
    main()