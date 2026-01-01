
# Week 4: Optimization & Architecture (Backprop Ninja & WaveNet)

> **"The details are not the details. They make the design."** -- Charles Eames

Welcome to Week 4. **This is the hardest week of the course.**

Up until now, we have treated `loss.backward()` as a magic button. This week, you will open the black box. You will manually derive and implement the backward pass for every single layer including the complex mathematics of Batch Normalization and Cross-Entropy. This is the rite of passage for every deep learning engineer.

Once you have mastered the gradients, we will change the **Architecture**. We will move from the "flat" MLP (where every character sees every other character equally) to a **Hierarchical Tree** structure (WaveNet), allowing our model to process much longer sequences efficiently.

## Learning Objectives

* **Master Manual Backprop:** Derive gradients for Tensors, Matrix Multiplications, and Normalization layers without Autograd.
* **Understand Shapes & Broadcasting:** Debug tensor mismatches during the backward pass (the #1 source of bugs in custom layers).
* **Build WaveNet:** Implement dilated convolutions (hierarchical processing) to handle long context windows.
* **Optimize Training:** Fix the "Hockey Stick" loss curve by proper initialization.

---

## The Lectures (Double Feature)

### Part 1: The Calculus (Backprop Ninja)
**Video:** [Building makemore Part 4: Becoming a Backprop Ninja](https://youtu.be/q8SA3rM6ckI)
* **Focus:** Manually implementing the `backward()` pass for every layer in our MLP.
* **Key Takeaway:** Understanding exactly how gradients flow through `view`, `reshape`, and `broadcasting` operations.

### Part 2: The Architecture (WaveNet)
**Video:** [Building makemore Part 5: Building a WaveNet](https://youtu.be/t3YqWQ66dQI)
* **Focus:** Scaling up! Moving from a context of 3 characters to 8 (or more) using a tree-like structure.
* **Key Concept:** **Dilated Convolutions.** Instead of crushing all inputs into one hidden layer, we fuse them progressively (2->1, 2->1), similar to how Byte Pair Encoding or CNNs work.

---

## Survival Guide & Workflow

This week is dense. Do not try to do everything at once. Follow this sequence to avoid burnout:

* **Step 1: Warm Up with Easy Layers.** Watch the first half of Video 4. Implement the backward pass for `Linear`, `Tanh`, and simple math ops. Get those `cmp()` checks to pass first.
* **Step 2: Tackle the Beast (BatchNorm).** Focus entirely on the backward pass for **Batch Normalization** and **Cross Entropy**. These are mathematically heavy. *Hint: Use pen and paper to derive the derivatives before coding.*
* **Step 3: Switch Gears to Architecture.** Watch Video 5. Forget about gradients for a moment and focus on **tensor shapes** (`B, T, C`). Build the WaveNet classes so the data flows correctly through the hierarchy.
* **Step 4: Train & Polish.** Train your WaveNet on `names.txt`. Compare the loss to your Week 3 MLP. Clean up your code for submission.

---

## Resources

### 1. Math Survival Kit (For Backprop Ninja)
* **[The Matrix Calculus You Need For Deep Learning](https://arxiv.org/abs/1802.01528)** (Parr & Howard) - **The Bible of Matrix Derivatives.** If you are confused about how to differentiate a matrix multiplication, read this.
* **[Yes you should understand backprop](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b)** (Andrej Karpathy) - An iconic blog post explaining *why* we are doing Part 1.
* **[Deriving the Gradient for Batch Normalization](https://kevinzakka.github.io/2016/09/14/batch_normalization/)** - A step-by-step derivation. **Extremely helpful** when you get stuck on the BatchNorm exercise.

### 2. Architecture Deep Dives (WaveNet)
* **[WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)** (DeepMind, 2016) - The inspiration for this week's architecture. *Read Figure 3 closely.*
* **[DeepMind's WaveNet Blog](https://www.deepmind.com/blog/wavenet-a-generative-model-for-raw-audio)** - Contains the famous animations showing how dilated convolutions "grow" their receptive field exponentially.
* **[Dilated Convolutions Explained](https://towardsdatascience.com/understanding-2d-dilated-convolution-operation-with-examples-in-numpy-and-pytorch-7d0a58e94439)** - Visual guide to "holes" in convolutions.

---

## The Assignment

**Goal:** Complete the "Backprop Ninja" challenge and replicate the WaveNet architecture.

### Step 1: Part 1 - Become a Backprop Ninja
**The Task:** Manually implement the backward pass for a deep network.

1.  **Open the Exercise:** [Click here to open the "Backprop Ninja" Colab](https://colab.research.google.com/drive/1WV2oi2fh9XXyldh02wupFQX0wh5ZC-z-?usp=sharing).
2.  **The Workflow:**
    * The notebook contains a massive chunk of code with the `backward` pass commented out.
    * **Don't watch the whole video at once.** Watch until Andrej explains a layer (e.g., `Linear`), pause, implement it yourself, and check with `cmp()`.
    * If you get stuck, unpause and watch him solve it. This is a learning exercise, not a test.
3.  **Pro Tips:**
    * **Summing Gradients:** When a tensor is used multiple times (bifurcation), its gradients must be *summed*.
    * **Broadcasting:** If a forward op involved broadcasting (e.g., adding a bias vector to a matrix), the backward op must involve a `sum()` to reduce the dimensions back.

### Step 2: Part 2 - Build WaveNet
**The Task:** Transform your flat MLP into a hierarchical Tree.

1.  Create a new notebook `week4_wavenet.ipynb` (or look at `makemore_wavenet.ipynb` for reference).
2.  **Implementation Strategy:**
    * **Flattening:** Understand how to use `tensor.view()` to treat the batch and time dimensions together when passing through dense layers.
    * **The Hierarchy:** Implement the "fusion" step. If you have 8 characters, layer 1 should produce 4 vectors, layer 2 should produce 2, and layer 3 should produce 1.
    * **Container:** Use `torch.nn.Sequential` to stack your custom blocks cleanly.
3.  **Goal:** Train this hierarchical model on `names.txt` and achieve a validation loss comparable to or better than your MLP from Week 3 (`< 2.2`), but with a larger context window (e.g., 8).

---

## Submission

1.  **Save your work:**
    * In Colab: `File` > `Save a copy in GitHub`.
    * Destination: Your **forked repository**.
    * Path: Save the Backprop notebook as `week4/backprop_ninja.ipynb` and the WaveNet notebook as `week4/wavenet.ipynb`.
    * Message: "Week 4: Conquered Backprop and WaveNet".
2.  **Check:** Ensure both files are present in your repo.

**Next Week:** The Grand Finale. We take everything we've learned - Embeddings, Normalization, Hierarchy, and build the **Transformer (GPT)**.
