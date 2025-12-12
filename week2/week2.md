
# Week 2: Language Modeling with Bigrams

> **"The limits of my language mean the limits of my world."** —> Ludwig Wittgenstein

Welcome to Week 2! Now that you understand the engine of learning (backpropagation), we will apply it to data.

This week, we start **Language Modeling**. You will build a model that learns to generate "names" by predicting the next character in a sequence. We will start with a simple statistical approach (counting) and then reproduce the exact same results using a **Neural Network**. This is your bridge from "pure math" to "Deep Learning with PyTorch."

##  Learning Objectives

* **Understand Language Modeling:** The core task of predicting the next token given a context.
* **Master PyTorch Tensors:** Moving from scalar `Value` objects to efficient, multi-dimensional `torch.Tensor` operations.
* **Broadcasting Semantics:** The most critical (and confusing) concept in vectorized programming.
* **Negative Log Likelihood (NLL):** The standard loss function for classification tasks.

---

##  The Lecture

**Video:** [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo)
* **Duration:** ~2 hours
* **Instructor:** Andrej Karpathy

**Advice:** This video introduces **PyTorch**. Pay extremely close attention to how `torch.Tensor` operations work, especially when dimensions don't match (broadcasting).

---

##  Star Resources

### 1. The Core Concepts
* **[A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)** (Bengio et al., 2003) - The academic paper that established the foundations we are implementing (though we start simpler than this).
* **[The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/)** (Jay Alammar) - Helps visualize how neural networks handle words/characters as vectors.

### 2. PyTorch Essentials (Crucial)
* **[PyTorch Tensor Tutorial](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html)** - Official guide to the `Tensor` object.
* **[Broadcasting Semantics](https://pytorch.org/docs/stable/notes/broadcasting.html)** - **Read this.** If you don't understand broadcasting, your tensor code will fail silently.

---

##  The Assignment

**Goal:** Understand the Bigram model and then upgrade it to a Trigram model.

### Step 1: The Lecture Code
1.  Navigate to the **`week2/`** folder in your forked repository.
2.  Open `makemore_part1_bigrams.ipynb`.
3.  This notebook contains the code written by Andrej in the video. **Run through it cell-by-cell.**
4.  Ensure you understand *why* the matrix shapes match and how `broadcasting` is used in lines like `P /= P.sum(1, keepdims=True)`.

### Step 2: The Exercises (Your Task)
At the bottom of the notebook (or in a new one), complete the following exercises to level up your model.

* **E01:** Train a **trigram** language model, i.e. take two characters as an input to predict the 3rd one. Feel free to use either counting or a neural net. Evaluate the loss; Did it improve over a bigram model?
* **E02:** Split up the dataset randomly into **80% train set, 10% dev set, 10% test set**. Train the bigram and trigram models only on the training set. Evaluate them on dev and test splits. What can you see?
* **E03:** Use the dev set to **tune the strength of smoothing** (or regularization) for the trigram model - i.e. try many possibilities and see which one works best based on the dev set loss. What patterns can you see in the train and dev set loss as you tune this strength? Take the best setting of the smoothing and evaluate on the test set once and at the end. How good of a loss do you achieve?
* **E04:** We saw that our 1-hot vectors merely select a row of `W`, so producing these vectors explicitly feels wasteful. Can you delete our use of `F.one_hot` in favor of simply **indexing into rows of W**?
* **E05:** Look up and use `F.cross_entropy` instead. You should achieve the same result. Can you think of why we'd prefer to use `F.cross_entropy` instead?
* **E06:** Meta-exercise! Think of a fun/interesting exercise and complete it. **This Exercise has a lot of weightage, so think of a good implementation**

---

##  Submission

1.  **Save your work:**
    * In Colab: `File` > `Save a copy in GitHub`.
    * Destination: Your **forked repository**.
    * Path: `week2/week2_assignment.ipynb`.
    * Commit message: "Week 2: Completed Trigram Exercises".
2.  **Check:** Ensure the file is updated in your repo.

**Next Week:** We scale up! We will replace the simple linear layer with a **Multi-Layer Perceptron (MLP)** and learn meaningful embeddings.
