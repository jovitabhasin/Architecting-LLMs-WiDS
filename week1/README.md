# Week 1: The Engine of AI (Autograd & Backpropagation)

> **"You cannot understand the neural network unless you build the neural network."**

Welcome to Week 1! We are skipping the high-level APIs (like `torch.nn` or `Keras`) and going straight to the metal.

This week, you will build **micrograd**: a tiny Autograd engine. You will implement the mathematical machinery that allows neural networks to learn. By the end of this week, the "magic" of how ChatGPT or Stable Diffusion updates its weights will be demystified it's just the Chain Rule, applied recursively.

## Learning Objectives

* **Master Derivatives:** Calculate gradients analytically and numerically.
* **Understand the Computation Graph:** How mathematical expressions are built as trees of operations.
* **Build the `Value` Object:** Extend the provided starter code to track history and gradients.
* **Implement Softmax & Loss:** Use your engine to calculate the standard classification loss.

---

## Optional: Python OOP Refresher

**Status:** *Optional (Skip if you are comfortable with Classes, `__init__`, and `__repr__`)*.

The core assignment relies heavily on Python's Object-Oriented Programming (OOP) features. If you need a refresher, these are the **Corey Schafer** tutorials referenced in the course PDF:

* **[Tutorial 1: Classes and Instances](https://www.youtube.com/watch?v=ZDa-Z5JzLYM)** - The basics of `class`, `self`, and `__init__`.
* **[Tutorial 2: Class Variables](https://www.youtube.com/watch?v=BJ-VvGyQxho)** - Sharing data between all instances.
* **[Tutorial 3: Class Methods and Static Methods](https://www.youtube.com/watch?v=rq8cL2XMM5M)** - The `@classmethod` and `@staticmethod` decorators.
* **[Tutorial 4: Inheritance - Creating Subclasses](https://www.youtube.com/watch?v=RSl87lqOXDE)** - How to extend classes (crucial for PyTorch later).
* **[Tutorial 5: Special (Magic/Dunder) Methods](https://www.youtube.com/watch?v=3ohzBxoFHAY)** - **Essential:** How to override `__add__` and `__mul__`.
* **[Tutorial 6: Property Decorators](https://www.youtube.com/watch?v=jCzT9XFZ5bw)** - Using `@property` for getters and setters.

---

## The Lecture

**Video:** [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)
* **Duration:** ~2 hours 30 mins
* **Instructor:** Andrej Karpathy

**Advice:** Do not just watch. **Type along.** Pause the video, implement the method, and inspect the variables.

---

## Star Resources (The "Why" & "How")

Before or during your coding, use these resources to visualize the math.

### 1. Visual Intuition (Highly Recommended)
* **[But what is a Neural Network?](https://www.youtube.com/watch?v=aircAruvnKk)** (3Blue1Brown) - The high-level view of layers and weights.
* **[Gradient Descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w)** (3Blue1Brown) - Visualizing the "cost landscape."
* **[What is backpropagation really doing?](https://www.youtube.com/watch?v=Ilg3gGewQ5U)** (3Blue1Brown) - The best visual explanation of the chain rule.

### 2. Reference Code
* **[micrograd GitHub Repo](https://github.com/karpathy/micrograd)** - Karpathy's original code. Use this to debug if you get stuck.

---

## The Assignment

**Goal:** Complete the **`micrograd_exercises.ipynb`** notebook found in the `week1/` folder.

### Step 1: Open the Notebook
1.  Navigate to the **`week1/`** folder in your forked repository.
2.  Open `micrograd_exercises.ipynb`.
3.  Click the **"Open in Colab"** badge (if available) or upload the file to [Google Colab](https://colab.research.google.com/).

### Step 2: Section 1 - Derivatives
The first half of the notebook focuses on pure calculus concepts to ensure you understand *what* a gradient is before you automate it.
* **Task 1:** Implement `gradf()` to return the **analytical gradient** (using calculus rules).
* **Task 2:** Implement the **numerical gradient** approximation (finite difference).
* **Task 3:** Implement the **symmetric derivative** approximation for better precision.

### Step 3: Section 2 - Support for Softmax
The notebook provides a **starter `Value` class** with basic functionality (`__init__`, `__repr__`, `__add__`). You must extend it to support the operations needed for the Softmax function and Cross-Entropy Loss.

**You need to implement/enable:**
* **Operations:** `exp()`, `log()`.
* **Magic Methods:** `__neg__` (negation), `__sub__` (subtraction), `__pow__` (power), `__truediv__` (division).
* **Reverse Ops:** Ensure `__radd__`, `__rmul__`, etc., work so `1 + x` works just like `x + 1`.

### Step 4: Verification
* **Sanity Check:** The notebook contains a cell that calculates `loss` using your `Value` class. If your gradients match the expected output `ans`, you are correct.
* **PyTorch Check:** The final cell asks you to implement the same logic in PyTorch to confirm the results match exactly.

---

## Submission

1.  **Save your work:**
    * In Colab, go to `File` > `Save a copy in GitHub`.
    * Select your **forked repository**.
    * Ensure the file path matches **`week1/micrograd_exercises.ipynb`** so it updates the existing file.
    * Commit message: "Week 1: Completed Micrograd Exercises".
2.  **Verify:** Check your forked repo on GitHub to ensure your changes are visible.

**Next Week:** We move from scalar values to **Tensors** and start building language models!


