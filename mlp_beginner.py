"""
Character-level MLP for name generation following industry best practices.

This module implements a character-level language model using PyTorch.

WHAT THIS DOES IN SIMPLE TERMS:
- Reads a list of names from a file
- Learns patterns in how letters follow each other
- Creates new names by predicting what letter comes next
- Like autocomplete, but for generating entire names!
"""

# Import all the tools we need
import logging  # For printing nice messages with timestamps
import random  # For shuffling data randomly
from dataclasses import dataclass  # For organizing settings in one place
from pathlib import Path  # For handling file paths
from typing import Dict, List, Tuple, Optional  # For making code clearer about what types of data we use

# PyTorch - the main AI library we're using
import torch  # Main library for AI computations (like NumPy but for AI)
import torch.nn as nn  # Pre-built AI building blocks (like LEGO pieces)
import torch.nn.functional as F  # Common AI functions (like loss calculations)
import torch.optim as optim  # Tools for teaching the AI (optimizers)
from torch.utils.data import DataLoader, Dataset, random_split  # Tools for handling data

# Set up logging - this makes our print statements look professional
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """All the settings for our AI model - like a recipe card!"""
    
    # HOW BIG IS OUR AI BRAIN?
    embedding_dim: int = 10  # How many numbers represent each letter (like 10 coordinates for each letter)
    context_window: int = 3  # How many previous letters the AI looks at (like looking at last 3 letters)
    hidden_size: int = 200  # How many "neurons" in the middle layer (bigger = smarter but slower)
    vocab_size: Optional[int] = None  # How many different letters we have (set automatically)
    
    # HOW DO WE TEACH THE AI?
    train_steps: int = 20000  # How many times we show examples to the AI (more = better learning)
    batch_size: int = 32  # How many examples we show at once (like studying 32 flashcards at once)
    learning_rate: float = 0.1  # How fast the AI learns (too fast = chaotic, too slow = boring)
    min_learning_rate: float = 0.01  # Slowest learning speed (we slow down as we get better)
    
    # HOW DO WE DIVIDE OUR DATA?
    train_ratio: float = 0.8  # Use 80% of names for teaching
    val_ratio: float = 0.9  # Use 10% for testing (remaining 10% unused)
    
    # SPECIAL SYMBOLS
    start_end_token: str = '.'  # Special symbol that means "start/end of name"
    start_token_index: int = 0  # Number we use for the special symbol
    
    # MAKING RESULTS REPEATABLE
    random_seed: int = 42  # Magic number to make random things predictable
    torch_seed: int = 2147483647  # Another magic number for PyTorch
    
    # HOW OFTEN TO PRINT PROGRESS
    log_interval: int = 1000  # Print progress every 1000 steps


class CharacterDataset(Dataset):
    """This class prepares our name data for the AI to learn from.
    
    Think of it like organizing flashcards for studying:
    - Each card shows 3 letters
    - The AI has to guess what the 4th letter should be
    """
    
    def __init__(self, words: List[str], vocab: Dict[str, int], context_window: int):
        # Store the vocabulary (which letter = which number)
        self.vocab = vocab
        # How many previous letters to look at
        self.context_window = context_window
        # Create all the training examples
        self.data = self._build_dataset(words)
    
    def _build_dataset(self, words: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create training examples from names.
        
        For each name like "Emma":
        1. Start with "..." (3 dots)
        2. Show "..." → predict "E" (first letter)
        3. Show "..E" → predict "m" (second letter)
        4. Show ".Em" → predict "m" (third letter)
        5. Show "Emm" → predict "a" (fourth letter)
        6. Show "mma" → predict "." (end of name)
        """
        inputs, targets = [], []  # Lists to store our training examples
        
        for word in words:
            # Start with dots: "..." (like starting with empty context)
            context = [self.vocab['.']] * self.context_window
            
            # Go through each letter in the name + the ending dot
            for char in word + '.':
                next_idx = self.vocab[char]  # Convert letter to number
                inputs.append(context.copy())  # Save the context (3 previous letters)
                targets.append(next_idx)  # Save what letter should come next
                # Move the window: drop oldest letter, add newest letter
                context = context[1:] + [next_idx]
        
        # Convert lists to PyTorch tensors (like converting to arrays for fast computation)
        return torch.tensor(inputs), torch.tensor(targets)
    
    def __len__(self) -> int:
        """How many training examples do we have?"""
        return len(self.data[0])
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get one training example by its number."""
        return self.data[0][idx], self.data[1][idx]


class CharacterMLP(nn.Module):
    """This is our AI brain! It learns to predict the next letter.
    
    Think of it like a very smart autocomplete:
    - You give it 3 letters
    - It tells you what letter should come next
    - It gets smarter by seeing thousands of examples
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # STEP 1: Convert letters to numbers (like a dictionary)
        # Each letter becomes a list of 10 numbers (like coordinates)
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        
        # STEP 2: The "thinking" part of our AI
        # We look at 3 letters × 10 numbers each = 30 numbers total
        flattened_dim = config.context_window * config.embedding_dim
        # First layer: 30 numbers → 200 numbers (the "hidden" thinking)
        self.fc1 = nn.Linear(flattened_dim, config.hidden_size, bias=False)
        # Normalize the numbers (like adjusting volume to be just right)
        self.ln1 = nn.LayerNorm(config.hidden_size)
        # Second layer: 200 numbers → how many letters we have (final decision)
        self.fc2 = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Set up the AI's "brain" with good starting values
        self._init_weights()
    
    def _init_weights(self):
        """Set up the AI's brain with good starting values.
        
        Like giving a student good study materials before they start learning!
        """
        # Set the first layer weights (the "thinking" part)
        # Xavier initialization: like giving balanced starting knowledge
        nn.init.xavier_normal_(self.fc1.weight, gain=5/3)
        
        # Set the output layer weights (the "decision making" part)
        # Start small so the AI doesn't make wild guesses at first
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.zeros_(self.fc2.bias)  # Start with no bias (neutral)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The AI makes a prediction! This is like the AI "thinking".
        
        Input: 3 letters (as numbers)
        Output: probabilities for what the next letter should be
        
        Think of it like this:
        1. Convert letters to coordinates
        2. Think about patterns (hidden layer)
        3. Make a decision about the next letter
        """
        # STEP 1: Convert letters to coordinates
        # Input: [batch_size, 3] (like [32, 3] for 32 examples)
        # Output: [batch_size, 3, 10] (each letter becomes 10 numbers)
        embedded = self.embedding(x)
        
        # STEP 2: Flatten and think
        # Take all 30 numbers (3 letters × 10 each) and put them in one row
        flattened = embedded.view(embedded.size(0), -1)
        
        # STEP 3: The "thinking" process
        # First layer: 30 numbers → 200 numbers (hidden thinking)
        # Normalize: adjust the numbers to be in a good range
        # Tanh: squish numbers between -1 and 1 (like adjusting volume)
        hidden = torch.tanh(self.ln1(self.fc1(flattened)))
        
        # STEP 4: Make the final decision
        # 200 numbers → probabilities for each possible letter
        logits = self.fc2(hidden)
        
        return logits
    
    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Vocabulary:
    """This class manages the "dictionary" between letters and numbers.
    
    Like a translator:
    - Computers only understand numbers
    - We need to convert letters to numbers and back
    - This class does that translation!
    """
    
    def __init__(self, words: List[str], start_end_token: str = '.'):
        self.start_end_token = start_end_token
        # Create the translation dictionaries
        self.char_to_idx, self.idx_to_char = self._build_vocab(words)
        self.vocab_size = len(self.char_to_idx)
    
    def _build_vocab(self, words: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build the letter-to-number dictionary from all the names."""
        # Find all unique letters in all names
        chars = sorted(set(''.join(words)))
        # Add our special "end of name" symbol first
        chars = [self.start_end_token] + chars
        
        # Create two dictionaries:
        # char_to_idx: 'a' → 1, 'b' → 2, etc.
        # idx_to_char: 1 → 'a', 2 → 'b', etc.
        char_to_idx = {char: idx for idx, char in enumerate(chars)}
        idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        
        return char_to_idx, idx_to_char
    
    def encode(self, text: str) -> List[int]:
        """Convert letters to numbers (like "abc" → [1, 2, 3])."""
        return [self.char_to_idx[char] for char in text]
    
    def decode(self, indices: List[int]) -> str:
        """Convert numbers back to letters (like [1, 2, 3] → "abc")."""
        return ''.join(self.idx_to_char[idx] for idx in indices)


def load_data(file_path: str) -> List[str]:
    """Read all the names from a file.
    
    Like opening a book and reading all the names on each line.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logger.error(f"Data file not found: {file_path}")
        raise


def setup_reproducibility(config: ModelConfig):
    """Make sure our results are repeatable.
    
    Like setting the same random seed in a video game:
    - Same seed = same random results every time
    - This helps us debug and compare results
    """
    random.seed(config.random_seed)  # Set Python's random number generator
    torch.manual_seed(config.torch_seed)  # Set PyTorch's random number generator
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.torch_seed)  # Set GPU random numbers too
        torch.cuda.manual_seed_all(config.torch_seed)


def main() -> None:
    """This is where everything happens! The main training process.
    
    Think of it like teaching a student:
    1. Get the study materials (names)
    2. Set up the classroom (model)
    3. Teach the student (training)
    4. Test the student (validation)
    5. See what the student learned (generate names)
    """
    # STEP 1: Set up our "recipe card" with all the settings
    config = ModelConfig()
    
    # STEP 2: Make sure we get the same results every time
    setup_reproducibility(config)
    
    # STEP 3: Load all the names from the file
    logger.info("Loading data...")
    words = load_data("names.txt")
    logger.info(f"Loaded {len(words)} names")
    
    # STEP 4: Create our letter-to-number translator
    vocab = Vocabulary(words, config.start_end_token)
    config.vocab_size = vocab.vocab_size
    logger.info(f"Vocabulary size: {vocab.vocab_size}")
    
    # STEP 5: Prepare the training data (like making flashcards)
    dataset = CharacterDataset(words, vocab.char_to_idx, config.context_window)
    
    # STEP 6: Split our data into training and testing sets
    # Like dividing flashcards: some for studying, some for testing
    train_size = int(config.train_ratio * len(dataset))
    val_size = int((config.val_ratio - config.train_ratio) * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.random_seed)
    )
    
    # STEP 7: Create data loaders (like organizing flashcards into batches)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # STEP 8: Create our AI brain
    model = CharacterMLP(config)
    logger.info(f"Model created with {model.get_num_parameters()} parameters")
    
    # STEP 9: Choose where to run (GPU if available, otherwise CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    model.to(device)  # Move the AI brain to the chosen device
    
    # STEP 10: Set up the "teacher" (optimizer and learning rate)
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.train_steps, eta_min=config.min_learning_rate
    )
    
    # STEP 11: THE TRAINING LOOP - This is where the magic happens!
    logger.info("Starting training...")
    model.train()  # Tell the AI it's time to learn
    step = 0
    
    while step < config.train_steps:
        # Go through all our training examples in small batches
        for batch_x, batch_y in train_loader:
            if step >= config.train_steps:
                break
                
            # Move data to the right device (GPU or CPU)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # TEACHING STEP 1: Show the AI some examples and see what it predicts
            logits = model(batch_x)  # AI makes predictions
            loss = F.cross_entropy(logits, batch_y)  # Calculate how wrong it was
            
            # TEACHING STEP 2: Tell the AI what it did wrong and how to improve
            optimizer.zero_grad()  # Clear old mistakes
            loss.backward()  # Calculate what to change
            optimizer.step()  # Make the changes
            scheduler.step()  # Adjust learning speed
            
            step += 1
            
            # Print progress so we know it's working
            if step % config.log_interval == 0 or step == 1:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Step {step}/{config.train_steps}: "
                          f"Loss={loss.item():.4f}, LR={current_lr:.6f}")
    
    # STEP 12: TEST THE AI - See how well it learned!
    logger.info("Training completed!")
    model.eval()  # Tell the AI it's time for a test (no more learning)
    with torch.no_grad():  # Don't calculate gradients during testing
        val_losses = []
        # Test the AI on examples it hasn't seen before
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            val_loss = F.cross_entropy(logits, batch_y)
            val_losses.append(val_loss.item())
        
        # Calculate average test score
        avg_val_loss = sum(val_losses) / len(val_losses)
        logger.info(f"Final validation loss: {avg_val_loss:.4f}")
    
    # STEP 13: SHOW OFF! Generate some new names to see what the AI learned
    logger.info("Generating sample names...")
    model.eval()  # Make sure we're in test mode
    with torch.no_grad():  # Don't learn from generating names
        for i in range(10):
            # Start with empty context (just dots)
            context = [vocab.char_to_idx['.']] * config.context_window
            generated_indices = []
            
            # Keep generating letters until we hit the "end" symbol
            for _ in range(20):  # Maximum length (safety limit)
                # Convert context to tensor and ask the AI for next letter
                context_tensor = torch.tensor(context).unsqueeze(0).to(device)
                
                # AI makes its prediction
                logits = model(context_tensor)
                probs = F.softmax(logits, dim=1).squeeze(0)  # Convert to probabilities
                
                # Pick the next letter (randomly, but weighted by AI's confidence)
                next_idx = torch.multinomial(probs, 1).item()
                
                # Stop if we hit the "end of name" symbol
                if next_idx == vocab.char_to_idx['.']:
                    break
                
                # Add this letter to our generated name
                generated_indices.append(next_idx)
                # Update context: drop oldest letter, add newest letter
                context = context[1:] + [next_idx]
            
            # Convert numbers back to letters and show the result
            name = vocab.decode(generated_indices)
            logger.info(f"  {i+1:2d}. {name}")


if __name__ == "__main__":
    main()