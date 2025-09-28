"""
Character-level MLP for name generation following industry best practices.

This module implements a character-level language model using PyTorch,
following patterns used by OpenAI, Anthropic, and other leading AI companies.
"""

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

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
    
    # Regularization
    dropout: float = 0.0
    weight_decay: float = 0.0
    
    # Special tokens
    start_end_token: str = '.'
    start_token_index: int = 0
    
    # Reproducibility
    random_seed: int = 42
    torch_seed: int = 2147483647
    
    # Logging and checkpointing
    log_interval: int = 1000
    save_interval: int = 5000
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"


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
    """
    Character-level MLP for name generation.
    
    Architecture:
    - Embedding layer
    - Linear layer + LayerNorm + Tanh
    - Output layer
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        
        # Hidden layers
        flattened_dim = config.context_window * config.embedding_dim
        self.fc1 = nn.Linear(flattened_dim, config.hidden_size, bias=False)
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using best practices."""
        # Xavier initialization for fc1
        nn.init.xavier_normal_(self.fc1.weight, gain=5/3)
        
        # Small initialization for output layer
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, context_window]
            
        Returns:
            Logits tensor of shape [batch_size, vocab_size]
        """
        # Embedding lookup
        embedded = self.embedding(x)  # [batch_size, context_window, embedding_dim]
        
        # Flatten context dimension
        flattened = embedded.view(embedded.size(0), -1)  # [batch_size, context_window * embedding_dim]
        
        # Hidden layer with normalization and activation
        hidden = torch.tanh(self.ln1(self.fc1(flattened)))
        hidden = self.dropout(hidden)
        
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


class Trainer:
    """Training class following industry best practices."""
    
    def __init__(self, model: CharacterMLP, config: ModelConfig, device: str = 'cpu'):
        self.model = model
        self.config = config
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Setup optimizer and scheduler
        self.optimizer = optim.SGD(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.train_steps, 
            eta_min=config.min_learning_rate
        )
        
        # Setup logging
        self.writer = SummaryWriter(config.log_dir)
        
        # Training state
        self.step = 0
        self.best_val_loss = float('inf')
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            # Forward pass
            logits = self.model(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.step += 1
            
            # Logging
            if self.step % self.config.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Step {self.step}/{self.config.train_steps}: "
                          f"Loss={loss.item():.4f}, LR={current_lr:.6f}")
                
                self.writer.add_scalar('Train/Loss', loss.item(), self.step)
                self.writer.add_scalar('Train/LearningRate', current_lr, self.step)
            
            # Checkpointing
            if self.step % self.config.save_interval == 0:
                self.save_checkpoint()
            
            # Early stopping if we've reached max steps
            if self.step >= self.config.train_steps:
                break
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def evaluate(self, val_loader: DataLoader) -> float:
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                logits = self.model(batch_x)
                loss = F.cross_entropy(logits, batch_y)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Log validation loss
        self.writer.add_scalar('Val/Loss', avg_loss, self.step)
        
        # Save best model
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint(is_best=True)
        
        return avg_loss
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_step_{self.step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with validation loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")


class NameGenerator:
    """Name generation utility."""
    
    def __init__(self, model: CharacterMLP, vocab: Vocabulary, device: str = 'cpu'):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.model.eval()
    
    def generate_name(self, max_length: int = 20, temperature: float = 1.0) -> str:
        """Generate a single name."""
        with torch.no_grad():
            # Start with context of start tokens
            context = [self.vocab.char_to_idx['.']] * self.model.config.context_window
            generated_indices = []
            
            for _ in range(max_length):
                # Convert to tensor
                context_tensor = torch.tensor(context).unsqueeze(0).to(self.device)
                
                # Forward pass
                logits = self.model(context_tensor)
                
                # Apply temperature and get probabilities
                logits = logits / temperature
                probs = F.softmax(logits, dim=1).squeeze(0)
                
                # Sample next character
                next_idx = torch.multinomial(probs, 1).item()
                
                # Stop if we hit the end token
                if next_idx == self.vocab.char_to_idx['.']:
                    break
                
                generated_indices.append(next_idx)
                context = context[1:] + [next_idx]
            
            return self.vocab.decode(generated_indices)
    
    def generate_names(self, num_names: int = 10, **kwargs) -> List[str]:
        """Generate multiple names."""
        return [self.generate_name(**kwargs) for _ in range(num_names)]


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


def main():
    """Main training function."""
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
    
    # Create trainer
    trainer = Trainer(model, config, device)
    
    # Training loop
    logger.info("Starting training...")
    while trainer.step < config.train_steps:
        # Train epoch
        train_loss = trainer.train_epoch(train_loader)
        
        # Evaluate
        val_loss = trainer.evaluate(val_loader)
        
        logger.info(f"Epoch completed - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Final evaluation
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    
    # Generate sample names
    generator = NameGenerator(model, vocab, device)
    sample_names = generator.generate_names(10)
    
    logger.info("Sample generated names:")
    for i, name in enumerate(sample_names, 1):
        logger.info(f"  {i:2d}. {name}")
    
    # Close tensorboard writer
    trainer.writer.close()


if __name__ == "__main__":
    main()
