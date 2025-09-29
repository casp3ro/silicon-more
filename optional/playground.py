"""
AI Name Generator Playground - A Beginner's Guide to Character-Level Language Models

This educational playground teaches you how artificial intelligence learns to generate 
human-like names by discovering patterns in character sequences. Perfect for beginners 
who want to understand machine learning fundamentals!

Learning Objectives:
- How computers convert text into numbers for processing
- Understanding character patterns and transitions in names
- How neural networks learn complex patterns automatically
- Comparing simple counting methods vs. intelligent learning approaches
- Generating new names using trained AI models

This tutorial builds understanding step-by-step, from basic concepts to advanced techniques.
"""

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random

# =============================================================================
# 🎨 VISUALIZATION HELPER FUNCTIONS
# =============================================================================

def show_character_transition_heatmap(transition_counts, char_to_index):
    """
    Visualize character transition patterns as a heatmap.
    
    This function creates a visual representation showing how often each character
    follows another character in our training data. Darker colors indicate more
    frequent transitions, helping us understand the patterns our AI needs to learn.
    """
    print("Creating character transition visualization...")
    
    # Set up the visualization
    plt.figure(figsize=(12, 12))
    plt.imshow(transition_counts, cmap="Blues")
    
    # Create character mapping for labels
    index_to_char = {i: char for char, i in char_to_index.items()}
    
    # Add detailed labels to each cell
    for i in range(transition_counts.shape[0]):
        for j in range(transition_counts.shape[1]):
            # Display character transition (e.g., "a" -> "b")
            char_pair = f"{index_to_char[i]}→{index_to_char[j]}"
            count = transition_counts[i, j].item()
            
            # Add transition labels and counts
            plt.text(j, i, char_pair, ha="center", va="bottom", 
                    color="gray", fontsize=8)
            plt.text(j, i, str(count), ha="center", va="top", 
                    color="black" if count > 0 else "lightgray", fontsize=6)
    
    plt.title("Character Transition Patterns\n(Frequency of character sequences)", 
              fontsize=14, pad=20)
    plt.xlabel("Next Character", fontsize=12)
    plt.ylabel("Current Character", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def demonstrate_one_hot_encoding():
    """
    Demonstrate how computers convert text characters into numerical representations.
    
    Computers can only work with numbers, not letters. One-hot encoding is a method
    that converts each character into a unique numerical "fingerprint" that the computer
    can process and understand.
    """
    print("\n" + "="*60)
    print("STEP 3: CONVERTING CHARACTERS TO NUMBERS")
    print("="*60)
    
    # Create a simple vocabulary for demonstration
    chars = ['.', 'a', 'b', 'c']
    char_to_idx = {char: i for i, char in enumerate(chars)}
    
    print(f"Vocabulary: {chars}")
    print(f"Character-to-number mapping: {char_to_idx}")
    print()
    
    print("One-hot encoding converts each character to a unique number pattern:")
    for char in chars:
        # Create a vector of zeros
        one_hot = torch.zeros(len(chars))
        # Set the position for this character to 1
        one_hot[char_to_idx[char]] = 1
        print(f"  '{char}' becomes: {one_hot.tolist()}")
    
    print("\nKey insight: Each character gets a unique 'fingerprint' of numbers.")
    print("This allows the computer to distinguish between different characters")
    print("and perform mathematical operations on text data.")

# =============================================================================
# 📚 STEP 1: UNDERSTANDING THE DATA
# =============================================================================

def explore_dataset(names):
    """
    Analyze the name dataset to understand its characteristics and patterns.
    
    Before training any AI model, we must first understand our data. This analysis
    helps us identify patterns, validate data quality, and set appropriate expectations
    for what our AI model can learn.
    """
    print("\n" + "="*60)
    print("STEP 1: ANALYZING THE NAME DATASET")
    print("="*60)
    
    print(f"Dataset Statistics:")
    print(f"  Total names: {len(names):,}")
    print(f"  Average length: {sum(len(name) for name in names) / len(names):.1f} characters")
    print(f"  Shortest name: '{min(names, key=len)}' ({len(min(names, key=len))} chars)")
    print(f"  Longest name: '{max(names, key=len)}' ({len(max(names, key=len))} chars)")
    
    # Display sample names
    print(f"\nSample names from the dataset:")
    for i, name in enumerate(names[:10]):
        print(f"  {i+1:2d}. {name}")
    
    # Show random sample
    print(f"\nRandom sample:")
    random_sample = random.sample(names, 10)
    for i, name in enumerate(random_sample):
        print(f"  {i+1:2d}. {name}")
    
    print(f"\nObservations about name patterns:")
    print(f"  • Names have consistent starting patterns")
    print(f"  • Common letter combinations appear frequently")
    print(f"  • Names follow predictable ending patterns")
    print(f"  • These patterns are what our AI will learn to replicate")

def build_character_vocabulary(names):
    """
    Create a character vocabulary and mapping system for the dataset.
    
    To process text with computers, we need to create a vocabulary that maps each
    unique character to a number. This includes all letters plus special tokens
    for marking the beginning and end of names.
    """
    print("\n" + "="*60)
    print("STEP 2: BUILDING CHARACTER VOCABULARY")
    print("="*60)
    
    # Collect all unique characters from the dataset
    all_chars = set()
    for name in names:
        all_chars.update(name)
    
    # Create ordered vocabulary with special start/end token
    chars = ['.'] + sorted(list(all_chars))
    
    # Create bidirectional mappings
    char_to_idx = {char: i for i, char in enumerate(chars)}
    idx_to_char = {i: char for i, char in enumerate(chars)}
    
    print(f"Vocabulary created:")
    print(f"  Total unique characters: {len(chars)}")
    print(f"  Characters: {chars}")
    print(f"\nCharacter-to-number mapping:")
    for char, idx in char_to_idx.items():
        print(f"  '{char}' → {idx}")
    
    print(f"\nSpecial tokens:")
    print(f"  '.' (period) marks the START and END of each name")
    print(f"  This helps the AI know when to begin and stop generating")
    
    return char_to_idx, idx_to_char, chars

# =============================================================================
# 📊 STEP 3: THE COUNTING APPROACH (BIGRAM MODEL)
# =============================================================================

def count_character_transitions(names, char_to_idx):
    """
    Count character transition frequencies using a simple statistical approach.
    
    This method counts how often each character follows another character in our
    training data. This creates a probability table that can be used to generate
    new names by following the most common patterns.
    """
    print("\n" + "="*60)
    print("STEP 4: COUNTING CHARACTER TRANSITIONS")
    print("="*60)
    
    # Initialize transition count matrix
    num_chars = len(char_to_idx)
    transition_counts = torch.zeros((num_chars, num_chars), dtype=torch.int32)
    
    print("Analyzing character sequences in training data...")
    total_transitions = 0
    
    # Count all character transitions
    for name in names:
        # Add start and end markers to each name
        chars_with_tokens = ['.'] + list(name) + ['.']
        
        # Count each consecutive character pair
        for i in range(len(chars_with_tokens) - 1):
            current_char = chars_with_tokens[i]
            next_char = chars_with_tokens[i + 1]
            
            current_idx = char_to_idx[current_char]
            next_idx = char_to_idx[next_char]
            
            transition_counts[current_idx, next_idx] += 1
            total_transitions += 1
    
    print(f"Analysis complete:")
    print(f"  Total character transitions analyzed: {total_transitions:,}")
    
    # Display most common transitions
    print(f"\nMost frequent character transitions:")
    flat_indices = torch.argsort(transition_counts.flatten(), descending=True)
    for i in range(10):
        idx = flat_indices[i].item()
        row, col = idx // num_chars, idx % num_chars
        count = transition_counts[row, col].item()
        if count > 0:
            current_char = list(char_to_idx.keys())[list(char_to_idx.values()).index(row)]
            next_char = list(char_to_idx.keys())[list(char_to_idx.values()).index(col)]
            print(f"  '{current_char}' → '{next_char}': {count:,} occurrences")
    
    return transition_counts

def convert_counts_to_probabilities(transition_counts, char_to_idx):
    """
    Convert transition counts into probability distributions.
    
    Raw counts are converted to probabilities that sum to 1.0 for each character.
    Laplace smoothing is applied to handle unseen character combinations gracefully.
    """
    print("\n" + "="*60)
    print("STEP 5: CONVERTING COUNTS TO PROBABILITIES")
    print("="*60)
    
    # Apply Laplace smoothing to avoid zero probabilities
    smoothed_counts = transition_counts + 1
    print("Applying Laplace smoothing (+1 to all counts)")
    print("This ensures every character combination has a small probability")
    
    # Convert to probabilities by normalizing each row
    probabilities = smoothed_counts.float()
    probabilities = probabilities / probabilities.sum(1, keepdim=True)
    
    print(f"\nProbability conversion complete:")
    print(f"  Each row now sums to 1.0 (100% probability)")
    print(f"  Ready for name generation using statistical sampling")
    
    # Display example probability distributions
    print(f"\nExample probability distributions:")
    char_names = ['.', 'a', 'e', 'n']  # Show examples for common characters
    for char in char_names:
        if char in char_to_idx:
            char_idx = char_to_idx[char]
            probs = probabilities[char_idx]
            print(f"\n  After '{char}':")
            # Show top 5 most likely next characters
            top_probs, top_indices = torch.topk(probs, 5)
            for prob, idx in zip(top_probs, top_indices):
                next_char = list(char_to_idx.keys())[list(char_to_idx.values()).index(idx.item())]
                print(f"    '{next_char}': {prob.item():.3f} ({prob.item()*100:.1f}%)")
    
    return probabilities

def generate_names_with_counting_model(probabilities, idx_to_char, num_names=10):
    """
    Generate names using the statistical counting model.
    
    This method uses the probability table to generate new names by sampling
    characters based on their learned frequencies. It demonstrates how
    simple statistical patterns can create new text.
    """
    print("\n" + "="*60)
    print("STEP 6: GENERATING NAMES WITH COUNTING MODEL")
    print("="*60)
    
    # Set random seed for reproducible results
    g = torch.Generator().manual_seed(42)
    
    print(f"Generating {num_names} names using statistical approach...")
    print("This model follows the most common character patterns from training data")
    
    generated_names = []
    for i in range(num_names):
        # Start with the beginning marker
        current_idx = 0  # '.' represents start/end
        name_chars = []
        max_length = 20  # Prevent infinite generation
        
        # Generate characters until we hit the end marker
        for _ in range(max_length):
            # Get probability distribution for next character
            next_char_probs = probabilities[current_idx]
            
            # Sample next character based on probabilities
            next_idx = torch.multinomial(next_char_probs, 1, generator=g).item()
            next_char = idx_to_char[next_idx]
            
            if next_idx == 0:  # Hit the end marker
                break
            
            name_chars.append(next_char)
            current_idx = next_idx
        
        generated_name = ''.join(name_chars)
        generated_names.append(generated_name)
        print(f"  {i+1:2d}. {generated_name}")
    
    print(f"\nStatistical model characteristics:")
    print(f"  • Follows common letter patterns from training data")
    print(f"  • May produce unrealistic combinations")
    print(f"  • Limited to simple character-pair patterns")
    print(f"  • Demonstrates basic pattern recognition")
    
    return generated_names

# =============================================================================
# 🧠 STEP 6: THE NEURAL NETWORK APPROACH
# =============================================================================

def demonstrate_neural_network_approach(names, char_to_idx, idx_to_char):
    """
    Prepare training data for the neural network approach.
    
    Neural networks learn by seeing many examples of input-output pairs.
    We create training examples where the input is a character and the output
    is the next character that should follow it.
    """
    print("\n" + "="*60)
    print("STEP 7: PREPARING NEURAL NETWORK TRAINING DATA")
    print("="*60)
    
    print("Creating training examples for neural network...")
    print("Each example: current character → next character")
    
    xs, ys = [], []
    
    # Use more names for better training results
    demo_names = names[:1000]  # Increased from 100 to 1000
    
    # Extract character transition examples
    for name in demo_names:
        chars_with_tokens = ['.'] + list(name) + ['.']
        for i in range(len(chars_with_tokens) - 1):
            current_char = chars_with_tokens[i]
            next_char = chars_with_tokens[i + 1]
            
            current_idx = char_to_idx[current_char]
            next_idx = char_to_idx[next_char]
            
            xs.append(current_idx)
            ys.append(next_idx)
    
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    
    print(f"Training data prepared:")
    print(f"  Total training examples: {len(xs):,}")
    print(f"  Training names used: {len(demo_names):,}")
    print(f"  Each example: current character → next character")
    
    # Show sample training examples
    print(f"\nSample training examples:")
    for i in range(10):
        current_char = idx_to_char[xs[i].item()]
        next_char = idx_to_char[ys[i].item()]
        print(f"  {i+1:2d}. '{current_char}' → '{next_char}'")
    
    return xs, ys

def train_simple_neural_network(xs, ys, char_to_idx, idx_to_char, num_steps=5000):
    """
    Train a very simple neural network to predict the next character.
    This demonstrates the core concepts of machine learning!
    """
    print(f"\n🎓 TRAINING A SIMPLE NEURAL NETWORK")
    print("=" * 50)
    
    num_chars = len(char_to_idx)
    
    # Initialize neural network weights (more powerful architecture)
    g = torch.Generator().manual_seed(42)
    
    # Two-layer network: input -> hidden -> output
    hidden_size = 64  # Hidden layer size
    W1 = torch.randn((num_chars, hidden_size), generator=g, requires_grad=True)
    W1.data *= torch.sqrt(torch.tensor(2.0 / num_chars))  # Xavier initialization
    
    W2 = torch.randn((hidden_size, num_chars), generator=g, requires_grad=True)
    W2.data *= torch.sqrt(torch.tensor(2.0 / hidden_size))  # Xavier initialization
    
    b2 = torch.zeros(num_chars, requires_grad=True)  # Bias for output layer
    
    print(f"🎲 Initialized neural network:")
    print(f"   Input layer: {num_chars} characters")
    print(f"   Hidden layer: {hidden_size} neurons")
    print(f"   Output layer: {num_chars} characters")
    print(f"💡 This is more powerful than a single linear layer!")
    print(f"💡 Using Xavier initialization for better training stability!")
    
    # Training loop
    print(f"\n🔄 Training for {num_steps} steps...")
    print(f"   (Each step: make prediction → calculate error → adjust weights)")
    
    for step in range(num_steps):
        # Clear gradients at the start of each step
        for param in [W1, W2, b2]:
            if param.grad is not None:
                param.grad.zero_()
        
        # Forward pass: make predictions
        xenc = F.one_hot(xs, num_classes=num_chars).float()  # Convert to one-hot
        
        # Two-layer forward pass
        hidden = torch.tanh(xenc @ W1)  # First layer with tanh activation
        logits = hidden @ W2 + b2  # Second layer with bias
        
        # Calculate loss using cross-entropy (more stable than manual softmax)
        loss = F.cross_entropy(logits, ys)
        
        # Backward pass: calculate gradients
        loss.backward()  # This computes gradients for all parameters
        
        # Update weights (this is the learning!)
        # Learning rate scheduling: start high, then decrease
        if step < num_steps // 3:
            learning_rate = 0.05  # Reduced from 0.1
        elif step < 2 * num_steps // 3:
            learning_rate = 0.01
        else:
            learning_rate = 0.005  # Even smaller for fine-tuning
            
        # Update all parameters
        with torch.no_grad():  # Don't track gradients during weight update
            W1.data -= learning_rate * W1.grad
            W2.data -= learning_rate * W2.grad
            b2.data -= learning_rate * b2.grad
        
        # Print progress
        if step % 500 == 0:  # Print less frequently since we have more steps
            print(f"   Step {step:4d}: Loss = {loss.item():.4f} (LR = {learning_rate})")
    
    print(f"✅ Training complete! Final loss: {loss.item():.4f}")
    print(f"💡 Lower loss = better predictions!")
    
    return W1, W2, b2

def generate_names_with_neural_network(W1, W2, b2, idx_to_char, num_names=10):
    """
    Generate names using our trained neural network.
    This should produce more realistic names than the counting approach!
    """
    print(f"\n🎭 GENERATING NAMES WITH NEURAL NETWORK")
    print("=" * 50)
    
    g = torch.Generator().manual_seed(123)
    
    print(f"🎲 Generating {num_names} names using neural network...")
    print(f"💡 This model learned complex patterns, not just simple counting!")
    
    generated_names = []
    for i in range(num_names):
        current_idx = 0  # Start with '.'
        name_chars = []
        max_length = 20  # Safety limit to prevent infinite loops
        
        # Generate characters until we hit the end token
        for _ in range(max_length):
            # One-hot encode current character
            xenc = F.one_hot(torch.tensor([current_idx]), num_classes=len(idx_to_char)).float()
            
            # Get neural network prediction (two-layer forward pass)
            hidden = torch.tanh(xenc @ W1)  # First layer with tanh activation
            logits = hidden @ W2 + b2  # Second layer with bias
            probs = F.softmax(logits, dim=1)  # Convert logits to probabilities
            
            # Sample next character
            next_idx = torch.multinomial(probs[0], 1, generator=g).item()
            next_char = idx_to_char[next_idx]
            
            if next_idx == 0:  # End token
                break
            
            name_chars.append(next_char)
            current_idx = next_idx
        
        generated_name = ''.join(name_chars)
        generated_names.append(generated_name)
        print(f"   {i+1:2d}. {generated_name}")
    
    print(f"\n🎉 Notice how these names:")
    print(f"   - Look more realistic and varied")
    print(f"   - Show complex patterns the network learned")
    print(f"   - Are more creative than simple counting!")
    
    return generated_names

# =============================================================================
# 🎯 MAIN FUNCTION - THE COMPLETE JOURNEY
# =============================================================================

def main():
    """
    Execute the complete AI name generation tutorial.
    
    This function demonstrates the entire process from data analysis to AI-generated
    names, comparing simple statistical methods with advanced neural network approaches.
    """
    print("="*70)
    print("AI NAME GENERATOR TUTORIAL")
    print("="*70)
    print("This tutorial teaches you how artificial intelligence learns to generate")
    print("human-like names by discovering patterns in character sequences.")
    print("We'll compare simple counting methods with neural network approaches.")
    print()
    
    # Load and validate the dataset
    print("Loading name dataset...")
    try:
        with open("names.txt", "r") as file:
            names = file.read().splitlines()
        
        # Validate dataset
        if not names:
            print("ERROR: names.txt is empty!")
            return
        
        # Clean and filter names
        names = [name.strip() for name in names if name.strip()]
        if not names:
            print("ERROR: No valid names found in names.txt!")
            return
            
        print(f"SUCCESS: Loaded {len(names):,} names")
        
    except FileNotFoundError:
        print("ERROR: names.txt file not found!")
        print("Please ensure names.txt exists in the same directory as this script.")
        return
    except Exception as e:
        print(f"ERROR loading names.txt: {e}")
        return
    
    explore_dataset(names)
    
    # Step 2: Build vocabulary
    char_to_idx, idx_to_char, chars = build_character_vocabulary(names)
    
    # Step 3: Demonstrate one-hot encoding
    demonstrate_one_hot_encoding()
    
    # Step 4: Counting approach
    transition_counts = count_character_transitions(names, char_to_idx)
    
    # Optional: Show the heatmap (uncomment to see it)
    # show_character_transition_heatmap(transition_counts, char_to_idx)
    
    probabilities = convert_counts_to_probabilities(transition_counts, char_to_idx)
    counting_names = generate_names_with_counting_model(probabilities, idx_to_char)
    
    # Step 5: Neural network approach
    xs, ys = demonstrate_neural_network_approach(names, char_to_idx, idx_to_char)
    W1, W2, b2 = train_simple_neural_network(xs, ys, char_to_idx, idx_to_char)
    neural_names = generate_names_with_neural_network(W1, W2, b2, idx_to_char)
    
    # Compare the two approaches
    print("\n" + "="*70)
    print("COMPARISON: STATISTICAL vs NEURAL NETWORK APPROACHES")
    print("="*70)
    
    print("Statistical Counting Model Results:")
    for i, name in enumerate(counting_names[:5]):
        print(f"  {i+1}. {name}")
    
    print("\nNeural Network Model Results:")
    for i, name in enumerate(neural_names[:5]):
        print(f"  {i+1}. {name}")
    
    print(f"\n" + "="*50)
    print("TUTORIAL COMPLETE")
    print("="*50)
    print("You have successfully learned the fundamentals of character-level language models!")
    print()
    print("Key Concepts Learned:")
    print("• Text preprocessing: Converting characters to numerical representations")
    print("• Statistical modeling: Using frequency counts to predict patterns")
    print("• Neural networks: Learning complex patterns through training")
    print("• Model comparison: Evaluating different approaches to the same problem")
    print("• Name generation: Creating new text based on learned patterns")
    print()
    print("Next Steps:")
    print("• Experiment with different training data sizes")
    print("• Try adjusting neural network architecture")
    print("• Explore more advanced techniques in mlp.py")

if __name__ == "__main__":
    main()
