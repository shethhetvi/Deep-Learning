import numpy as np
import matplotlib.pyplot as plt
iterations = 20
eta = 0.1
beta1 = 0.9
beta2 = 0.9
beta2_adam = 0.999
weight_decay = 0.01
epsilon = 1e-8
initial_weight = 10.0
gradients = [1.0]
for i in range(1, iterations):
    gradients.append(max(0.1, gradients[-1] - 0.2))

# Separate lookahead-gradient schedule for NAG
nag_gradients = [1.0]
for i in range(1, iterations):
    nag_gradients.append(
        max(0.05, nag_gradients[-1] - 0.2)
    )
# Store weight trajectories
weights = {
    name: [initial_weight]
    for name in [
        "SGD", "Momentum", "NAG", "AdaGrad",
        "RMSProp", "Adam", "AdamW"
    ]
}
# Optimizer states
velocity_momentum = 0.5
velocity_nag = 0.5
G = 12.0
rms_variance = 12.0
adam_m = 0.5
adam_v = 12.0
# Run the 20 iterations
for i in range(iterations):
    g = gradients[i]
    g_nag = nag_gradients[i]
    # 1. SGD
    w = weights["SGD"][-1]
    weights["SGD"].append(w - eta * g)
    # 2. SGD with Momentum
    velocity_momentum = beta1 * velocity_momentum + g
    w = weights["Momentum"][-1]
    weights["Momentum"].append(
        w - eta * velocity_momentum
    )
    # 3. Nesterov Accelerated Gradient
    velocity_nag = beta1 * velocity_nag + g_nag
    w = weights["NAG"][-1]
    weights["NAG"].append(w - eta * velocity_nag)
    # 4. AdaGrad
    G += g ** 2
    w = weights["AdaGrad"][-1]
    update = eta * g / (np.sqrt(G) + epsilon)
    weights["AdaGrad"].append(w - update)
    # 5. RMSProp
    rms_variance = (
        beta2 * rms_variance + (1 - beta2) * g ** 2
    )
    w = weights["RMSProp"][-1]
    update = eta * g / (np.sqrt(rms_variance) + epsilon)
    weights["RMSProp"].append(w - update)
    # 6. Adam (without bias correction, as assumed
    # in the uploaded mathematical practical)
    adam_m = beta1 * adam_m + (1 - beta1) * g
    adam_v = (
        beta2_adam * adam_v
        + (1 - beta2_adam) * g ** 2
    )
    w = weights["Adam"][-1]
    update = eta * adam_m / (np.sqrt(adam_v) + epsilon)
    weights["Adam"].append(w - update)
    # 7. AdamW: Adam update followed by decoupled decay
    w = weights["AdamW"][-1]
    adamw_update = eta * adam_m / (
        np.sqrt(adam_v) + epsilon
    )
    w_adam = w - adamw_update
    w_adamw = w_adam - eta * weight_decay * w
    weights["AdamW"].append(w_adamw)
# Calculate absolute weight changes per iteration
weight_updates = {}
for name, trajectory in weights.items():
    weight_updates[name] = np.abs(np.diff(trajectory))
# Graph 1: Weight trajectories
plt.figure(figsize=(12, 6))
for name, trajectory in weights.items():
    plt.plot(
        range(iterations + 1),
        trajectory,
        marker="o",
        markersize=3,
        label=name
    )
plt.title("Weight Trajectories Across 20 Iterations")
plt.xlabel("Iteration")
plt.ylabel("Weight Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("optimizer_weight_trajectories.png", dpi=300)
plt.show()
# Graph 2: Weight updates
plt.figure(figsize=(12, 6))
for name, updates in weight_updates.items():
    plt.plot(
        range(1, iterations + 1),
        updates,
        marker="o",
        markersize=3,
        label=name
    )
plt.title("Weight Updates per Iteration")
plt.xlabel("Iteration")
plt.ylabel("Absolute Weight Change")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("optimizer_weight_updates.png", dpi=300)
plt.show()
print("\nFinal weights after", iterations, "iterations:")
for name, trajectory in weights.items():
    print(f"{name:10s}: {trajectory[-1]:.6f}")
