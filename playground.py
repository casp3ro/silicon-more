import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F  # tensor ops; we'll use one-hot and simple softmax-like normalization
import torch.nn.functional as F


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

    # # 4) Count character transitions into N
    # for name in names:
    #     chars = ['.'] + list(name) + ['.']  # Add start/end tokens around each name
    #     for ch1, ch2 in zip(chars, chars[1:]):  # Look at each pair of consecutive characters
    #         N[stoi[ch1], stoi[ch2]] += 1  # Increment count for this transition

    # # Optional: visualize the transition matrix
    # # show_heatmap(N, itos)

    # # 5) Fix random seed for reproducibility
    # g = torch.Generator().manual_seed(2147483647)
    
    # # 6) Convert counts to probabilities (Laplace smoothing by +1 to avoid zeros)
    # P = (N+1).float()
    # P /= P.sum(1, keepdim=True)  # Row-normalize so each row sums to 1
    

    # # 7) Sample names using the bigram model
    # for i in range(20):
    #     ix = 0  # Start with '.' (index 0)
    #     out = []  # Store generated characters

    #     # Generate characters until end token '.' (index 0) is sampled
    #     while True:
    #         p = P[ix]  # Get probability distribution for current character
    #         # Sample next character based on probabilities
    #         ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    #         out.append(itos[ix])  # Add the sampled character to output
    #         if ix == 0:  # If we sampled '.' (end token), stop generating
    #             break
    #     print(''.join(out))  # Print the complete generated name


    # # 8) Compute simple log-likelihood diagnostics on a few names
    # log_likelihood = 0.0
    # n = 0
    # for name in names[:3]:
    #     chars = ['.'] + list(name) + ['.']  
    #     for ch1, ch2 in zip(chars, chars[1:]): 
    #         ix = stoi[ch1]
    #         ix2 = stoi[ch2]
    #         prob = P[ix, ix2]
    #         logprob = torch.log(prob)
    #         log_likelihood += logprob
    #         n+=1
    #         print(f" {ch1} -> {ch2} : {logprob}")
    
    # print(f"Log likelihood: {log_likelihood}")
    # nll = -log_likelihood
    # print(f"Negative log likelihood: {nll}")
    # print(f"Average Negative log likelihood: {nll/n}")

    # 9) Build a small (x,y) dataset of next-character indices for the first 3 names
    #    xs holds current-character indices, ys holds the next-character indices.

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
    # xs: tensor of current-character indices; ys: tensor of next-character indices
    # Both have length equal to the number of adjacent character pairs extracted above
    print(xs)
    print(ys)

    # plt.imshow(xenc)
    # plt.show()
    # print(xenc.shape)
    
    # 10) Tiny linear bigram model (no training):
    #     - One-hot encode inputs
    #     - Apply linear layer W to get logits (unnormalized scores)
    #     - Exponentiate to get positive counts
    #     - Normalize rows to get probabilities (softmax without calling softmax)

    # RNDOMLY INTIALIZE
    g = torch.Generator().manual_seed(2147483647 +1)
    # W: linear layer weights mapping current-char (27-dim one-hot) -> next-char scores (27-dim)
    W = torch.randn((27,27), generator=g)
    # One-hot encode inputs so each row is a 27-dim indicator for the current char

    # FORWARD PASS
    xenc = F.one_hot(xs, num_classes=27).float()  # shape: [num_pairs, 27]
    # Linear scores (logits) for each possible next character
    logits = xenc @ W  # shape: [num_pairs, 27]
    # Convert scores to positive values; this is like unnormalized probabilities
    counts = logits.exp()
    # Normalize each row to sum to 1 to obtain a probability distribution (softmax-like)
    probs = counts / counts.sum(1, keepdim=True)
    # Negative log-likelihood (cross-entropy without reduction helpers):
    # pick the probability of the true next char for each row, take log, average negative
    loss = -probs[torch.arange(xs.size(0)), ys].log().mean()
    print(loss.item())

    # BACKWARD PASS
    W.grad = None
    loss.backward()
    print(W.grad)

    # UPDATE
    W.data += -0.1 * W.grad

    # print(probs)  # per-example distributions over next character

    # nlls = torch.zeros(5)

    # for i in range(5):
    #     x = xs[i].item()
    #     y = ys[i].item()
    #     p = probs[i,y]
    #     logp = torch.log(p)
    #     nll =-logp
    #     nlls[i] = nll
    
    # print(nlls.mean().item())


if __name__ == "__main__":
    main()