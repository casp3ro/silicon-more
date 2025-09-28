"""
Character-level MLP for name generation following industry best practices.

This module implements a character-level language model using PyTorch.
"""

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for the CharacterMLP model."""
    # Model architecture
    embedding_dim: int = 10
    context_window: int = 3
    hidden_size: int = 200
    vocab_size: Optional[int] = None  # Will be set based on data
    
    # Training hyperparameters
    train_steps: int = 20000
    batch_size: int = 32
    learning_rate: float = 0.1
    min_learning_rate: float = 0.01
    
    # Data splits
    train_ratio: float = 0.8
    val_ratio: float = 0.9
    
    # Special tokens
    start_end_token: str = '.'
    start_token_index: int = 0
    
    # Reproducibility
    random_seed: int = 42
    torch_seed: int = 2147483647
    
    # Logging
    log_interval: int = 1000


class CharacterDataset(Dataset):
    """Dataset class for character-level language modeling."""
    
    def __init__(self, words: List[str], vocab: Dict[str, int], context_window: int):
        self.vocab = vocab
        self.context_window = context_window
        self.data = self._build_dataset(words)
    
    def _build_dataset(self, words: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build (context, next_char) pairs using sliding window."""
        inputs, targets = [], []
        
        for word in words:
            # Initialize context with start tokens
            context = [self.vocab['.']] * self.context_window
            
            # Add start token and process each character
            for char in word + '.':
                next_idx = self.vocab[char]
                inputs.append(context.copy())
                targets.append(next_idx)
                # Slide window
                context = context[1:] + [next_idx]
        
        return torch.tensor(inputs), torch.tensor(targets)
    
    def __len__(self) -> int:
        return len(self.data[0])
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[0][idx], self.data[1][idx]


class CharacterMLP(nn.Module):
    """Character-level MLP for name generation using PyTorch built-in modules."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embedding layer: maps character indices to dense vectors
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        
        # Linear layers with proper initialization
        flattened_dim = config.context_window * config.embedding_dim
        self.fc1 = nn.Linear(flattened_dim, config.hidden_size, bias=False)
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Initialize weights using Xavier/Glorot initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using PyTorch best practices."""
        # Xavier initialization for fc1 (similar to original scaling)
        nn.init.xavier_normal_(self.fc1.weight, gain=5/3)
        
        # Small random initialization for fc2 (similar to original)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, context_window] containing character indices
            
        Returns:
            Logits tensor of shape [batch_size, vocab_size]
        """
        # Embedding lookup: [batch_size, context_window] -> [batch_size, context_window, embedding_dim]
        embedded = self.embedding(x)
        
        # Flatten context dimension: [batch_size, context_window, embedding_dim] -> [batch_size, context_window * embedding_dim]
        flattened = embedded.view(embedded.size(0), -1)
        
        # First linear layer + LayerNorm + tanh
        hidden = torch.tanh(self.ln1(self.fc1(flattened)))
        
        # Output layer
        logits = self.fc2(hidden)
        
        return logits
    
    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Vocabulary:
    """Vocabulary management for character-level modeling."""
    
    def __init__(self, words: List[str], start_end_token: str = '.'):
        self.start_end_token = start_end_token
        self.char_to_idx, self.idx_to_char = self._build_vocab(words)
        self.vocab_size = len(self.char_to_idx)
    
    def _build_vocab(self, words: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build character vocabulary from words."""
        # Get unique characters
        chars = sorted(set(''.join(words)))
        chars = [self.start_end_token] + chars  # Add special token first
        
        # Create mappings
        char_to_idx = {char: idx for idx, char in enumerate(chars)}
        idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        
        return char_to_idx, idx_to_char
    
    def encode(self, text: str) -> List[int]:
        """Encode text to indices."""
        return [self.char_to_idx[char] for char in text]
    
    def decode(self, indices: List[int]) -> str:
        """Decode indices to text."""
        return ''.join(self.idx_to_char[idx] for idx in indices)


def load_data(file_path: str) -> List[str]:
    """Load names from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logger.error(f"Data file not found: {file_path}")
        raise


def setup_reproducibility(config: ModelConfig):
    """Setup random seeds for reproducibility."""
    random.seed(config.random_seed)
    torch.manual_seed(config.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.torch_seed)
        torch.cuda.manual_seed_all(config.torch_seed)


def main() -> None:
    """Main training function following industry best practices."""
    # Configuration
    config = ModelConfig()
    
    # Setup reproducibility
    setup_reproducibility(config)
    
    # Load data
    logger.info("Loading data...")
    words = load_data("names.txt")
    logger.info(f"Loaded {len(words)} names")
    
    # Build vocabulary
    vocab = Vocabulary(words, config.start_end_token)
    config.vocab_size = vocab.vocab_size
    logger.info(f"Vocabulary size: {vocab.vocab_size}")
    
    # Create datasets
    dataset = CharacterDataset(words, vocab.char_to_idx, config.context_window)
    
    # Split data
    train_size = int(config.train_ratio * len(dataset))
    val_size = int((config.val_ratio - config.train_ratio) * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.random_seed)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Create model
    model = CharacterMLP(config)
    logger.info(f"Model created with {model.get_num_parameters()} parameters")
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.train_steps, eta_min=config.min_learning_rate
    )
    
    # Training loop
    logger.info("Starting training...")
    model.train()
    step = 0
    
    while step < config.train_steps:
        for batch_x, batch_y in train_loader:
            if step >= config.train_steps:
                break
                
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Forward pass
            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            step += 1
            
            # Logging
            if step % config.log_interval == 0 or step == 1:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Step {step}/{config.train_steps}: "
                          f"Loss={loss.item():.4f}, LR={current_lr:.6f}")
    
    # Final evaluation
    logger.info("Training completed!")
    model.eval()
    with torch.no_grad():
        val_losses = []
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            val_loss = F.cross_entropy(logits, batch_y)
            val_losses.append(val_loss.item())
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        logger.info(f"Final validation loss: {avg_val_loss:.4f}")
    
    # Generate sample names
    logger.info("Generating sample names...")
    model.eval()
    with torch.no_grad():
        for i in range(10):
            # Start with context of start tokens
            context = [vocab.char_to_idx['.']] * config.context_window
            generated_indices = []
            
            for _ in range(20):  # max length
                # Convert to tensor
                context_tensor = torch.tensor(context).unsqueeze(0).to(device)
                
                # Forward pass
                logits = model(context_tensor)
                probs = F.softmax(logits, dim=1).squeeze(0)
                
                # Sample next character
                next_idx = torch.multinomial(probs, 1).item()
                
                # Stop if we hit the end token
                if next_idx == vocab.char_to_idx['.']:
                    break
                
                generated_indices.append(next_idx)
                context = context[1:] + [next_idx]
            
            name = vocab.decode(generated_indices)
            logger.info(f"  {i+1:2d}. {name}")


if __name__ == "__main__":
    main()