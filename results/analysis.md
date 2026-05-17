### Gradient Conflict Analysis

The gradient conflict monitor shows how task gradients interact during training. When cosine similarity drops below zero, the shared backbone experienced a direct conflict between the two task objectives.

### Shared Representation Analysis

The shared representation inspector uses UMAP to project the backbone features into 2D. Coloring by Task A and Task B labels reveals whether the shared features support both tasks simultaneously.

### Final Performance Comparison

This section summarizes the final validation metrics for both the baseline and PCGrad models. It compares task-level accuracy and F1 score to evaluate whether gradient surgery improved multi-task learning stability.
