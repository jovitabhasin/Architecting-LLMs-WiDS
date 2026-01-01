
# Week 3: Deep Learning Internals (MLPs & Batch Normalization)

> **"We don't know why it works, but it works."** — *Common Deep Learning sentiment (which we will try to fix this week).*

Welcome to Week 3! This is a pivotal week where we move from simple architectures to the messy reality of training deep networks.

We have a **double feature** this week.
1.  First, we implement the **Multi-Layer Perceptron (MLP)** from the seminal Bengio et al. (2003) paper.
2.  Second, we dive into the "internals" to fix the classic demons of deep learning: vanishing gradients, saturated activations ("dead neurons"), and extreme sensitivity to initialization. We will implement **Batch Normalization** to stabilize training.

## 🎯 Learning Objectives

* **Build an MLP:** Implement a hidden layer with non-linearities (`tanh`) and Embeddings.
* **Master Internals:** Visualize histograms of activations and gradients to diagnose "dead" neurons.
* **Fix Training:** Implement **Kaiming Initialization** and **Batch Normalization** from scratch.
* **Optimization:** Understand learning rate tuning, train/dev/test splits, and under/overfitting.

---

## 📺 The Lectures (Double Feature)

### Part 1: The Architecture (MLP)
**Video:** [Building makemore Part 2: MLP](https://youtu.be/TCH_1BHY58I)
* **Focus:** Implementing the Bengio et al. 2003 paper. Moving from `count` based models to `embedding` based models. We cover model training, learning rate tuning, hyperparameters, evaluation, and splits.

### Part 2: The Internals (BatchNorm)
**Video:** [Building makemore Part 3: BatchNorm](https://youtu.be/P6sfmUTpUmc)
* **Focus:** Scrutinizing the statistics of forward pass activations and backward pass gradients. We learn why training deep nets is fragile and introduce **Batch Normalization** to make it easier.

---

## 📚 Star Resources

### 1. The Essential Papers
* **[A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)** (Bengio et al., 2003) - The architecture we are building in Part 1.
* **[Batch Normalization: Accelerating Deep Network Training](https://arxiv.org/abs/1502.03167)** (Ioffe & Szegedy, 2015) - The paper that introduced the most important layer in modern DL.
* **[Delving Deep into Rectifiers (Kaiming Init)](https://arxiv.org/abs/1502.01852)** (He et al., 2015) - Explains *how* to initialize weights so they don't vanish or explode.

### 2. Must-Read Blogs & Guides
* **[A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/)** (Karpathy) - **Crucial.** A practical guide on how to debug networks when the loss isn't going down.
* **[CS231n Notes: Neural Nets Part 2](https://cs231n.github.io/neural-networks-2/)** (Stanford) - Excellent written explanation of Batch Norm, Dropout, and Initialization.
* **[Understanding the Backward Pass through Batch Norm](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html)** - If you are struggling with the manual backprop exercise for BatchNorm, this step-by-step derivation is a lifesaver.

### 3. Visualizations & Intuition
* **[Visualizing the Loss Landscape of Neural Nets](https://losslandscapes.com/)** - See how "Skip Connections" and "Batch Norm" smooth out the terrain, making it easier for the optimizer to find the bottom.
* **[Kaiming Initialization Explanation](https://pouannes.github.io/blog/initialization/)** - A clear blog post explaining the math behind why we multiply weights by $\sqrt{2/n}$.

---

## 💻 The Assignment

**Goal:** Complete the exercises for both the MLP and BatchNorm videos.

### Step 1: Setup
1.  Navigate to the **`week3/`** folder in your fork.
2.  Create a notebook `week3_exercises.ipynb`.
3.  You can start with the code provided in the [Video 3 Notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part2_mlp.ipynb) and [Video 4 Notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part3_bn.ipynb).

### Step 2: Part 1 Exercises (MLP)
* **E01:** Tune the hyperparameters of the training to beat Andrej's best validation loss of **2.2**.
* **E02:** Initialization matters.
    1.  What is the loss you'd get if the predicted probabilities at initialization were perfectly uniform? What loss do we actually achieve?
    2.  Can you tune the initialization to get a starting loss that is much more similar to the uniform loss?
* **E03:** Read the **Bengio et al. 2003 paper** (link above). Implement and try any idea from the paper. Did it work?

### Step 3: Part 2 Exercises (Internals & BatchNorm)
* **E01:** I did not get around to seeing what happens when you **initialize all weights and biases to zero**. Try this and train the neural net.
    * You might think either that 1) the network trains just fine or 2) the network doesn't train at all, but actually it is 3) the network trains but only partially, and achieves a pretty bad final performance.
    * **Task:** Inspect the gradients and activations to figure out what is happening and why the network is only partially training, and what part is being trained exactly.
* **E02:** BatchNorm has the big advantage that after training, the parameters (gamma/beta) can be "folded into" the weights of the preceding Linear layers, erasing the need to forward it at test time.
    * **Task:** Set up a small 3-layer MLP with batchnorms, train the network, then "fold" the batchnorm gamma/beta into the preceding Linear layer's `W,b` by creating a new `W2, b2` and erasing the batch norm. Verify that this gives the same forward pass during inference.

---

## 🚀 Submission

1.  **Save your work:**
    * In Colab: `File` > `Save a copy in GitHub`.
    * Destination: Your **forked repository**.
    * Path: `week3/week3_exercises.ipynb`.
    * Message: "Week 3: Completed MLP and BatchNorm Exercises".
2.  **Check:** Ensure the file is updated in your repo.

**Next Week:** We leave the "fixed window" approach behind and build **WaveNets** and Recurrent structures!
