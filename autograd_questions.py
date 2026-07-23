"""Assignment 4, Part 3: gradient tracking experiments (Questions 6-9).

Run with:
    docker run --rm -v "$(pwd):/workspace" -w /workspace sps-genai \
        /code/.venv/bin/python autograd_questions.py
"""

import torch


def banner(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


print("torch version:", torch.__version__)


# ----------------------------------------------------------------------
banner("Question 6 (a): y = x^2 + 3x at x = 2, expected dy/dx = 2x + 3 = 7")

x = torch.tensor([2.0], requires_grad=True)
y = x**2 + 3 * x
y.backward()
print("x.grad =", x.grad)


# ----------------------------------------------------------------------
banner("Question 6 (b): same computation with requires_grad=False")

x = torch.tensor([2.0], requires_grad=False)
y = x**2 + 3 * x
print("y.requires_grad =", y.requires_grad)
print("y.grad_fn =", y.grad_fn)
try:
    y.backward()
except RuntimeError as error:
    print("RuntimeError:", error)
print("x.grad =", x.grad)


# ----------------------------------------------------------------------
banner("Question 6 (c) / 7 (c): default value of requires_grad")

x = torch.tensor([2.0])
print("torch.tensor([2.0]).requires_grad =", x.requires_grad)


# ----------------------------------------------------------------------
banner("Question 7 (a): w created without requires_grad")

x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([1.0, 3.0])
y = w[0] * x**2 + w[1] * x
y.backward()
print("x.grad =", x.grad)
print("w.grad =", w.grad)


# ----------------------------------------------------------------------
banner("Question 7 (b): w with requires_grad=True")
# Expected: dy/dx = 2*w0*x + w1 = 7, dy/dw0 = x^2 = 4, dy/dw1 = x = 2

x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([1.0, 3.0], requires_grad=True)
y = w[0] * x**2 + w[1] * x
y.backward()
print("x.grad =", x.grad)
print("w.grad =", w.grad)


# ----------------------------------------------------------------------
banner("Question 8: detach breaks the graph")

x = torch.tensor([1.0], requires_grad=True)
y = x * 3
z = y.detach()
print("z.requires_grad =", z.requires_grad)
print("z.grad_fn =", z.grad_fn)
w = z * 2
try:
    w.backward()
except RuntimeError as error:
    print("RuntimeError:", error)


banner("Question 8 (fixed): clone keeps z in the graph, expected dw/dx = 6")

x = torch.tensor([1.0], requires_grad=True)
y = x * 3
z = y.clone()
print("z.requires_grad =", z.requires_grad)
w = z * 2
w.backward()
print("x.grad =", x.grad)


# ----------------------------------------------------------------------
banner("Question 9: gradients accumulate across backward calls")

x = torch.tensor([1.0], requires_grad=True)
y1 = x * 2
y1.backward()
print("After first backward:  x.grad =", x.grad)
y2 = x * 3
y2.backward()
print("After second backward: x.grad =", x.grad, "(2 + 3 = 5, accumulated)")


banner("Question 9 (fixed): clear the buffer between backward calls")

x = torch.tensor([1.0], requires_grad=True)
y1 = x * 2
y1.backward()
print("After first backward:  x.grad =", x.grad)
x.grad = None  # same role as optimizer.zero_grad() in a training loop
y2 = x * 3
y2.backward()
print("After second backward: x.grad =", x.grad, "(no accumulation)")