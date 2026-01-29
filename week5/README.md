
# Week 5: The Grand Finale (Building GPT)

> **"The best way to predict the future is to invent it."** — Alan Kay

Welcome to the final week.

This is it. You have built the engine (Backprop), understood the fuel (Embeddings), and mastered the chassis (WaveNet/Hierarchy). Now, we replace the engine with a warp drive.

In this final week, you will build the **Transformer** architecture from scratch. You will implement the mechanism that changed the world **Self-Attention** and assemble these blocks into a functional Generative Pre-trained Transformer (GPT). By the end of this week, you will have a mini-ChatGPT running on your laptop that writes Shakespeare.

## Learning Objectives

* **Master Self-Attention:** Implement the `Key`, `Query`, and `Value` mechanism that allows the model to "route" information between tokens.
* **The Transformer Block:** Assemble Multi-Head Attention, Feed-Forward Networks, Residual connections, and LayerNorm into a single, stackable unit.
* **Positional Encodings:** Learn how to inject order information into a parallelized architecture.
* **Tokenizer (Optional):** Understand how raw text is converted into integers using Byte Pair Encoding (BPE).

---

## The Lectures

### Part 1: The Main Event (Building GPT)
**Video:** [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)
* **Duration:** ~2 hours
* **The Mission:** We start with an empty file and end with a GPT model. We build the `Head`, `MultiHeadAttention`, `FeedForward`, and `Block` classes one by one.
* **The Result:** A model that generates infinite Shakespeare-like text.

### Part 2: The Deep Dive (Optional)
**Video:** [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
* **Duration:** ~2 hours
* **Why watch?** In the main video, we use a simple character-level tokenizer. Real LLMs use **Byte Pair Encoding (BPE)**. This video explains the weirdness of LLMs (why they can't do math, why they are bad at spelling) by looking at how they "see" text.

---

## Star Resources

### 1. Code Repositories
* **[NanoGPT Repository](https://github.com/karpathy/nanogpt)** - The clean, professional reference implementation of what we are building.
* **[MinBPE Repository](https://github.com/karpathy/minbpe)** - The minimal, clean code for the Byte Pair Encoding tokenizer (Video 2).

### 2. The Papers (The Source of Truth)
* **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** (Vaswani et al., 2017) - The paper that started it all. *Read Section 3 (Model Architecture).*
* **[Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)** (OpenAI, 2020) - See how the architecture you are building scales up to 175 billion parameters.
* **[Introducing ChatGPT](https://openai.com/blog/chatgpt/)** (OpenAI Blog) - The blog post that launched the RLHF-tuned model to the world.

### 3. Compute & Hardware Advice
* **[Google Colab](https://colab.research.google.com/)** - The easiest way to get started (Free Tier includes T4 GPUs).
* **[Lambda Labs](https://lambdalabs.com/)** - If you want to train larger models seriously, this is often the cheapest/easiest cloud provider for on-demand high-end GPUs (A100/H100) compared to AWS/GCP.

---

## The Assignment

**Goal:** Build a functional GPT and complete the challenge exercises.

### Step 1: Build the Architecture (Video 1)
Follow the "Building GPT" video to implement the components in PyTorch:
1.  **The Head:** Implement `scaled_dot_product_attention`.
2.  **Multi-Head Attention:** Run multiple heads in parallel and concatenate their outputs.
3.  **The Block:** Combine Attention + FeedForward + LayerNorm + Residual Connections.
4.  **Training:** Train on the `tinyshakespeare` dataset until your loss drops below 2.0.

### Step 2: The Challenges (Exercises)
Once your base model is working, tackle these exercises to prove your mastery.

* **EX1: Tensor Mastery:** Combine the `Head` and `MultiHeadAttention` into one class that processes all the heads in parallel, treating the heads as another batch dimension. (Answer is in the [NanoGPT repo](https://github.com/karpathy/nanogpt)).
* **EX2: Custom Data / Calculator:** Train the GPT on your own dataset of choice!
    * *Advanced Suggestion:* Train a GPT to do addition (`a+b=c`).
    * *Tip:* Predict digits in reverse order (right to left).
    * *Tip:* Mask out the loss for the input positions (`a+b`) so the model only learns to predict the answer.
    * *Swole Doge Project:* Build a full calculator clone (+-*/). You may need Chain of Thought traces.
* **EX3: Pretraining & Finetuning:** Find a very large dataset (so large you can't see a gap between train and val loss). Pretrain the transformer on this data, then initialize with that model and finetune it on `tinyshakespeare` with fewer steps and a lower learning rate. Can you obtain a lower validation loss?
* **EX4: The Researcher:** Read some Transformer papers, pick **one** additional feature or architectural change (e.g., Rotary Embeddings, SwiGLU), and implement it. Does it improve performance?

### Step 3: (Optional) The Tokenizer
If you choose to do the Tokenizer deep dive:
1.  **Advised Flow:** Reference the [Google Colab from the video](https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L?usp=sharing) and try to implement the steps **before** Andrej gives away the partial solutions in the video.
2.  **Solutions:** If you get stuck, the full solution is in the [minbpe code](https://github.com/karpathy/minbpe/blob/master/minbpe/base.py).

---

## Final Submission

Congratulations! You have gone from a blank Python file to a functioning GPT.

1.  **Save your work:**
    * Save your notebook as `week5/gpt.ipynb`.
    * Include a sample of your generated text at the bottom.
2.  **Commit & Push:**
    * Message: "Week 5: I built GPT."
3.  **Celebrate:** You now understand the machinery behind the AI revolution.

---

## Where to Go from Here?

You have built the engine. If you want to continue your journey into AI Engineering and Research, here is your roadmap:

### 1. Scaling Up (Distributed Training)
* **Concept:** How do you train across multiple GPUs?
* **Resource:** Learn **PyTorch DDP (DistributedDataParallel)** and **FSDP (Fully Sharded Data Parallel)**.
* **Project:** Try to train your NanoGPT on 2 GPUs using `torch.distributed`.

### 2. Fine-Tuning & Alignment (RLHF)
* **Concept:** How do we turn a "next token predictor" into a helpful assistant?
* **Resource:** Read about **InstructGPT** and **RLHF (Reinforcement Learning from Human Feedback)**.
* **Tool:** Explore the **HuggingFace TRL** (Transformer Reinforcement Learning) library.

### 3. Efficiency & Inference
* **Concept:** How do we make these huge models run fast on small devices?
* **Resource:** Learn about **Quantization** (running models in 4-bit or 8-bit integers) and **FlashAttention**.
* **Code:** Look into `llama.cpp` to see how C++ is used for high-performance inference.

> **"What I cannot create, I do not understand."** — Richard Feynman
