import torch
import random

# Generate some noisy data points for fitting a line
def generate_data(num_points, noise_stddev=0.1):
    x = torch.linspace(0, 10, num_points)
    y = 2 * x + 1 + torch.randn(num_points) * noise_stddev
    return x, y

# Define a simple linear model as a PyTorch module
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.a = torch.nn.Parameter(torch.randn(1))
        self.b = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.a * x + self.b

# Define the loss function (Mean Squared Error)
def loss_fn(y_pred, y_true):
    return torch.mean((y_pred - y_true)**2)

# Differentiable RANSAC Algorithm
def differentiable_ransac(x, y, num_iterations, inlier_threshold):
    best_model = LinearModel()
    best_loss = float('inf')

    for _ in range(num_iterations):
        optimizer = torch.optim.SGD(best_model.parameters(), lr=0.01)

        # Randomly sample two data points
        sample_indices = random.sample(range(len(x)), 2)
        x_sample = x[sample_indices]
        y_sample = y[sample_indices]

        # Fit a model to the sampled points (in this case, a line)
        for _ in range(100):  # Inner optimization loop
            optimizer.zero_grad()
            y_pred = best_model(x_sample)
            loss = loss_fn(y_pred, y_sample)
            loss.backward()
            optimizer.step()

        # Calculate inliers based on a threshold
        y_pred = best_model(x)
        inliers = torch.abs(y_pred - y) < inlier_threshold

        # Calculate the loss for the inliers
        inlier_loss = loss_fn(y_pred[inliers], y[inliers])

        # Update the best model if this model has fewer inlier errors
        if inlier_loss < best_loss:
            best_loss = inlier_loss

    return best_model

# Generate noisy data
x, y = generate_data(100)

# Set RANSAC hyperparameters
num_iterations = 100
inlier_threshold = 0.2

# Run differentiable RANSAC
best_model = differentiable_ransac(x, y, num_iterations, inlier_threshold)

# Print the results
a = best_model.a.item()
b = best_model.b.item()
print("Differentiable RANSAC Model Parameters (a, b):", a, b)
