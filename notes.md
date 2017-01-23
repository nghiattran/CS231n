## Loss function

### SVM

Given an example (x<sub>i</sub>, y<sub>i</sub>) where x<sub>i</sub> is
the image, y<sub>i</sub> is the integer  label, and `s` =
f(x<sub>i</sub>, W)

The SVM loss has the form:

L<sub>i</sub>=∑<sub>j≠i</sub>max(0,s<sub>j</sub>−s<sub>yj</sub>+Δ)

and the full training loss is the mean over all examples in the training
data:

L = 1 / N * ∑<sub>j to N</sub>L<sub>i</sub>

**Regularization**. There is one bug with the loss function we presented
above because there are multiple set `W` that will correctly classify
all examples. Hence, we need an extra value to weight those solutions called
**regularization penalty** `R(W)`.

So the training loss full becomes:

L = 1 / N * ∑<sub>j to N</sub>L<sub>i</sub> + λR(W)

Where:

* `λ`: regulation strength (hyperparameter)


Common regulations:

* **L2 regulation**: R(W) = ∑<sub>k</sub>∑<sub>l</sub>(W<sup>2</sup><sub>k,l</sub>)
* **L1 regulation**: R(W) = ∑<sub>k</sub>∑<sub>l</sub>(|W<sub>k,l</sub>|)
* **Elastic net** (L1 + L2):  R(W) = ∑<sub>k</sub>∑<sub>l</sub>(βW<sup>2</sup><sub>k,l</sub> +
|W<sub>k,l</sub>|)
* **Max norm**:
* **Dropout**

### Softmax

**Probabilistic interpretation**:

With: `f` = f(x<sub>i</sub>, W)

P(y<sub>i</sub>∣x<sub>i</sub>)=e<sup>`f<sub>yi</sub></sup> / (∑<sub>j</sub>e<sup>f<sub>j</sub></sup>)

Where the right hand side is **softmax function**

We want to maximize the log likelihood or to minimize the negative log likelihood of the
correct class. (**cross-entropy loss** form)

L<sub>i</sub>=−log(P(y<sub>i</sub>∣x<sub>i</sub>))

Hence:

L<sub>i</sub>=−log(e<sup>f<sub>yi</sub></sup> / (∑<sub>j</sub>e<sup>f<sub>j</sub></sup>))

![SVM vs Softmax](svm-softmax.png)

### Optimization strategies

**Numerical gradient**: calculate gradient for every weight and do update.
Approximate, slow, but easy to write.

**Analytic gradient**: exact, fast but error-prone

Always use analytic gradient, but check implementation with numerical gradient
(**gradient check**)

### Mini-batch

Only use a small portion of the training set to compute the gradient

Common mini-batch sizes are 32/64/128..

