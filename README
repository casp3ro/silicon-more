## Silicon More — AI Name Generator (Beginner Friendly)

This small project shows **how AI can learn to generate human‑like names** from a simple text file.  
You do **not** need a math or machine‑learning background; everything is explained in plain language.

The core idea:  
We give the computer a list of real names (for example, California baby names).  
It studies the **patterns of letters** inside those names and then learns to **invent new names** that look similar.

---

### What’s in this folder (which files matter)

- **`names.txt`**  
  The only data file you need. It should contain **one name per line** (e.g. a list of California names).

- **`mlp_beginner.py` — Recommended starting point**  
  A **beginner‑friendly script** with lots of comments.  
  It trains a small AI model that reads `names.txt` and then generates new names.

- **`mlp_refactored.py` — Same idea, more “pro” style**  
  The same name‑generation model, but written in a more “industry” way: separate classes, logging, checkpoints, etc.  
  Use this after you are comfortable with `mlp_beginner.py`.

- **`optional/` (extra learning, can be ignored at first)**  
  - `playground.py`: a step‑by‑step tutorial that explains ideas like counting character pairs, one‑hot encoding, and a tiny neural network.
  - `silicon-more.py`: a very simple “bigram” model (it only looks at pairs of characters). Good for understanding the most basic version.

You can safely **ignore everything except `names.txt` and `mlp_beginner.py`** for your first run.

---

### Quick start (no ML knowledge required)

1. **Install Python**  
   Make sure you have **Python 3.9+** installed.

2. **Install PyTorch (the only heavy dependency)**  
   - On CPU only (simplest):
     ```bash
     pip install torch --index-url https://download.pytorch.org/whl/cpu
     ```
   - If you have a CUDA GPU and know what that is:
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```

3. **Put your names into `names.txt`**  
   - Each line = one name, for example:
     ```text
     Mia
     Olivia
     Noah
     Ethan
     Sofia
     ```
   - You can use any list of names (e.g. California baby names), not just English.

4. **Run the beginner‑friendly model**
   ```bash
   python mlp_beginner.py
   ```

5. **Watch the output**
   - You’ll see progress logs like:
     - “Loading data…”
     - “Step 1000/20000: Loss=2.05, LR=0.095”
   - At the end it will print **generated sample names**:
     ```text
     1. Kariel
     2. Alenna
     ...
     ```

That’s it — you’ve trained an AI and generated new names.

---

### What is happening conceptually? (plain language)

- **Input**: A simple text file with names: `names.txt`
- **Goal**: Learn “how letters usually follow each other” inside those names
- **Training**:
  - The model looks at a few previous letters (for example, `ann`)  
  - It predicts the next letter (for example, `a`)
  - If it’s wrong, it adjusts its internal numbers a tiny bit
  - Repeating this thousands of times makes it better and better
- **Generation**:
  - Start from a special “start” symbol
  - Predict the next letter, then the next, and so on
  - Stop when the model predicts the “end” symbol → that sequence of letters is one new name

You can think of it as **autocomplete for names**: instead of finishing one word, it invents entire names.

---

### Which script should I use?

- **If you’re new to AI / ML**  
  - Use **`mlp_beginner.py`** first.  
  - Ignore the `optional/` folder and `mlp_refactored.py` for now.

- **If you want to see “professional” structure**  
  - Read **`mlp_refactored.py`**.  
  - It uses classes like `Trainer`, `NameGenerator`, and better logging/checkpointing.

- **If you want a slow, guided tutorial**  
  - Open `optional/playground.py`.  
  - It walks you through:
    - Looking at the dataset
    - Counting how often one letter follows another
    - Building a basic counting model
    - Training a very small neural network
    - Comparing both approaches

- **If you want the absolute simplest math version**  
  - Run `optional/silicon-more.py`  
  - It trains a tiny **bigram model** (only pairs of letters) and generates sample words.

---

### How this project is organized (for non‑experts)

The project is intentionally split into:

- **Core scripts** (what you actually run)
  - `mlp_beginner.py`
  - `mlp_refactored.py` (same idea, more advanced layout)

- **Learning / optional scripts**
  - `optional/playground.py`
  - `optional/silicon-more.py`

- **Documentation**
  - `README` (this file): **start here**
  - `README_mlp.md`: extra details about the `mlp_beginner.py` script
  - `README_pytorch.md`: a short “PyTorch in plain English” guide

You can delete or ignore the `optional/` folder if you only care about **using** the model, not learning how it works.

---

### Next steps and simple experiments

Once the basic run works, try:

- **Change the dataset**:  
  Replace `names.txt` with:
  - Names from another country
  - City names
  - Product name ideas

- **Make the model bigger or smaller** (in `mlp_beginner.py`):  
  - Increase `hidden_size` to make it more expressive (but slower)  
  - Increase `train_steps` to let it learn longer

