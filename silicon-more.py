import torch
import matplotlib.pyplot as plt


def show_heatmap(N, itos):
    """Display the character transition count matrix as a heatmap with labels."""
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
    # 1) Load dataset (one name per line)
    with open("names.txt", "r") as file:
        names = file.read().splitlines()

    # 2) Prepare count matrix N for transitions (27 tokens: '.' + 26 letters)
    N = torch.zeros((27,27), dtype=torch.int32)

    # 3) Build vocabulary and index mappings
    chars = sorted(list(set(''.join(names))))
    chars = ['.'] + chars  # Add '.' at the beginning for start/end tokens
    stoi = {s:i for i,s in enumerate(chars)}  # String to index mapping
    itos = {i:s for i,s in enumerate(chars)}  # Index to string mapping

    # 4) Count character transitions into N
    for name in names:
        chars = ['.'] + list(name) + ['.']  # Add start/end tokens around each name
        for ch1, ch2 in zip(chars, chars[1:]):  # Look at each pair of consecutive characters
            N[stoi[ch1], stoi[ch2]] += 1  # Increment count for this transition

    # Optional: visualize the transition matrix
    # show_heatmap(N, itos)

    # 5) Fix random seed for reproducibility
    g = torch.Generator().manual_seed(2147483647)
    
    # 6) Convert counts to probabilities (Laplace smoothing by +1 to avoid zeros)
    P = (N+1).float()
    P /= P.sum(1, keepdim=True)  # Row-normalize so each row sums to 1
    

    # 7) Sample names using the bigram model
    for i in range(20):
        ix = 0  # Start with '.' (index 0)
        out = []  # Store generated characters

        # Generate characters until end token '.' (index 0) is sampled
        while True:
            p = P[ix]  # Get probability distribution for current character
            # Sample next character based on probabilities
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[ix])  # Add the sampled character to output
            if ix == 0:  # If we sampled '.' (end token), stop generating
                break
        print(''.join(out))  # Print the complete generated name


    # 8) Compute simple log-likelihood diagnostics on a few names
    log_likelihood = 0.0
    n = 0
    for name in names[:3]:
        chars = ['.'] + list(name) + ['.']  
        for ch1, ch2 in zip(chars, chars[1:]): 
            ix = stoi[ch1]
            ix2 = stoi[ch2]
            prob = P[ix, ix2]
            logprob = torch.log(prob)
            log_likelihood += logprob
            n+=1
            print(f" {ch1} -> {ch2} : {logprob}")
    
    print(f"Log likelihood: {log_likelihood}")
    nll = -log_likelihood
    print(f"Negative log likelihood: {nll}")
    print(f"Average Negative log likelihood: {nll/n}")

    # 9) Build a small (x,y) dataset of next-character indices for the first 3 names

    xs, ys = [], []

    for name in names[:3]:
        chars = ['.'] + list(name) + ['.']  
        for ch1, ch2 in zip(chars, chars[1:]): 
            ix = stoi[ch1]
            ix2 = stoi[ch2]
            print(f" {ch1} -> {ch2} : {ix} {ix2}")
            xs.append(ix)
            ys.append(ix2)
    
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    print(xs.shape, ys.shape)




if __name__ == "__main__":
    main()