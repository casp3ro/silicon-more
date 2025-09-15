import torch
import matplotlib.pyplot as plt


def show_heatmap(N, itos):
    """Display the character transition matrix as a heatmap with labels"""
    plt.figure(figsize=(16,16))
    plt.imshow(N, cmap="Blues")  # Show matrix as blue heatmap
    
    # Add character pair labels and counts to each cell
    for i in range(27):
        for j in range(27):
            chstr = itos[i] + itos[j]  # Character pair (e.g., "ab", ".e")
            plt.text(j, i, chstr, ha="center", va="bottom", color="gray")  # Character pair at bottom
            plt.text(j, i, str(N[i,j].item()), ha="center", va="top", color="gray")  # Count at top
    plt.axis("off")
    plt.show()


def main():
    # Load all names from the text file
    with open("names.txt", "r") as file:
        names = file.read().splitlines()

    # Create a 27x27 matrix to count character transitions (26 letters + '.' for start/end)
    N = torch.zeros((27,27), dtype=torch.int32)

    # Build character vocabulary: get all unique characters from names
    chars = sorted(list(set(''.join(names))))
    chars = ['.'] + chars  # Add '.' at the beginning for start/end tokens
    stoi = {s:i for i,s in enumerate(chars)}  # String to index mapping
    itos = {i:s for i,s in enumerate(chars)}  # Index to string mapping

    # Count character transitions: for each name, count how often each character follows another
    for name in names:
        chars = ['.'] + list(name) + ['.']  # Add start/end tokens around each name
        for ch1, ch2 in zip(chars, chars[1:]):  # Look at each pair of consecutive characters
            N[stoi[ch1], stoi[ch2]] += 1  # Increment count for this transition

    # Visualize the transition matrix
    # show_heatmap(N, itos)

    # Set random seed for reproducible name generation
    g = torch.Generator().manual_seed(2147483647)
    
    # Convert counts to probabilities: normalize each row to sum to 1
    P = N.float()  # Convert to float for division
    P /= P.sum(1, keepdim=True)  # Each row now sums to 1 (probability distribution)
    

    # Generate 20 new names using the learned character transition probabilities
    for i in range(20):
        ix = 0  # Start with '.' (index 0)
        out = []  # Store generated characters

        # Generate characters one by one until we hit the end token
        while True:
            p = P[ix]  # Get probability distribution for current character
            # Sample next character based on probabilities
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[ix])  # Add the sampled character to output
            if ix == 0:  # If we sampled '.' (end token), stop generating
                break
        print(''.join(out))  # Print the complete generated name


if __name__ == "__main__":
    main()