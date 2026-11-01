"""Experiment 02: Stateful PyTorch nn.Module vs JAX Pure Functions & PyTrees.

Compares in-place parameter mutation in PyTorch (`zero_grad`, `backward`, `step`)
against functional state transitions in JAX using a custom NamedTuple PyTree,
`jax.value_and_grad`, and `jax.tree.map`. Verifies step-by-step parity.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch import nn


class LinearBlockParams(NamedTuple):
    """Immutable PyTree holding two-layer MLP parameters."""

    w1: jax.Array
    b1: jax.Array
    w2: jax.Array
    b2: jax.Array


class TrainState(NamedTuple):
    """Explicit functional training state (parameters + SGD momentum velocity)."""

    params: LinearBlockParams
    velocity: LinearBlockParams
    step: jax.Array


class TorchMLP(nn.Module):
    """Equivalent stateful two-layer MLP in PyTorch."""

    def __init__(self, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.from_numpy(w1.copy()))
        self.b1 = nn.Parameter(torch.from_numpy(b1.copy()))
        self.w2 = nn.Parameter(torch.from_numpy(w2.copy()))
        self.b2 = nn.Parameter(torch.from_numpy(b2.copy()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(x @ self.w1 + self.b1)
        return h @ self.w2 + self.b2


def jax_mlp_forward(params: LinearBlockParams, x: jax.Array) -> jax.Array:
    """Pure forward pass with zero hidden state."""
    h = jnp.tanh(x @ params.w1 + params.b1)
    return h @ params.w2 + params.b2


def jax_mse_loss(params: LinearBlockParams, x: jax.Array, y: jax.Array) -> jax.Array:
    """Pure MSE loss function differentiated with respect to slot 0 (params)."""
    preds = jax_mlp_forward(params, x)
    return jnp.mean((preds - y) ** 2)


@jax.jit
def jax_train_step(
    state: TrainState, x: jax.Array, y: jax.Array, lr: float = 0.05, momentum: float = 0.9
) -> tuple[TrainState, jax.Array]:
    """Pure functional training step returning a brand-new TrainState PyTree."""
    loss, grads = jax.value_and_grad(jax_mse_loss)(state.params, x, y)
    new_velocity = jax.tree.map(lambda v, g: momentum * v + g, state.velocity, grads)
    new_params = jax.tree.map(lambda p, v: p - lr * v, state.params, new_velocity)
    new_state = TrainState(params=new_params, velocity=new_velocity, step=state.step + 1)
    return new_state, loss


def run_experiment(n_steps: int = 10) -> dict[str, Any]:
    """Compare PyTorch stateful training loop vs JAX pure PyTree transitions."""
    rng = np.random.default_rng(7)
    in_dim, hidden_dim, out_dim = 16, 32, 4
    w1_np = (rng.standard_normal((in_dim, hidden_dim)) * 0.1).astype(np.float32)
    b1_np = np.zeros((hidden_dim,), dtype=np.float32)
    w2_np = (rng.standard_normal((hidden_dim, out_dim)) * 0.1).astype(np.float32)
    b2_np = np.zeros((out_dim,), dtype=np.float32)

    x_np = rng.standard_normal((64, in_dim)).astype(np.float32)
    y_np = rng.standard_normal((64, out_dim)).astype(np.float32)

    # 1. Stateful PyTorch execution
    torch_model = TorchMLP(w1_np, b1_np, w2_np, b2_np)
    optimizer = torch.optim.SGD(torch_model.parameters(), lr=0.05, momentum=0.9)
    x_t = torch.from_numpy(x_np)
    y_t = torch.from_numpy(y_np)

    torch_losses: list[float] = []
    for _ in range(n_steps):
        optimizer.zero_grad()
        preds = torch_model(x_t)
        loss_t = torch.mean((preds - y_t) ** 2)
        loss_t.backward()
        optimizer.step()
        torch_losses.append(float(loss_t.item()))

    # 2. Pure functional JAX PyTree execution
    init_params = LinearBlockParams(
        w1=jnp.asarray(w1_np),
        b1=jnp.asarray(b1_np),
        w2=jnp.asarray(w2_np),
        b2=jnp.asarray(b2_np),
    )
    zero_velocity = jax.tree.map(jnp.zeros_like, init_params)
    state = TrainState(params=init_params, velocity=zero_velocity, step=jnp.array(0, dtype=jnp.int32))
    initial_w1_snapshot = np.asarray(state.params.w1).copy()

    x_j = jnp.asarray(x_np)
    y_j = jnp.asarray(y_np)

    jax_losses: list[float] = []
    for _ in range(n_steps):
        state, loss_j = jax_train_step(state, x_j, y_j)
        jax_losses.append(float(loss_j))

    # Verify original init_params was never mutated in-place
    init_params_unchanged = bool(np.array_equal(np.asarray(init_params.w1), initial_w1_snapshot))

    max_loss_diff = float(np.max(np.abs(np.array(torch_losses) - np.array(jax_losses))))
    max_w1_diff = float(
        np.max(np.abs(torch_model.w1.detach().numpy() - np.asarray(state.params.w1)))
    )
    leaves, treedef = jax.tree.flatten(state)

    return {
        "n_steps": n_steps,
        "pytree_num_leaves": len(leaves),
        "treedef_str": str(treedef),
        "initial_loss_torch": round(torch_losses[0], 6),
        "final_loss_torch": round(torch_losses[-1], 6),
        "initial_loss_jax": round(jax_losses[0], 6),
        "final_loss_jax": round(jax_losses[-1], 6),
        "max_loss_diff": max_loss_diff,
        "max_w1_diff": max_w1_diff,
        "init_params_unchanged": init_params_unchanged,
    }


def main() -> None:
    res = run_experiment()
    print("=" * 72)
    print("Experiment 02: Pure Functions & PyTrees vs Stateful nn.Module")
    print("=" * 72)
    print(f"PyTree Leaf Count       : {res['pytree_num_leaves']} leaves (params + velocity + step)")
    print(f"Init Params Unmutated   : {res['init_params_unchanged']} (zero in-place side effects)")
    print(f"PyTorch Loss (0 -> {res['n_steps']}) : {res['initial_loss_torch']:.6f} -> {res['final_loss_torch']:.6f}")
    print(f"JAX Loss     (0 -> {res['n_steps']}) : {res['initial_loss_jax']:.6f} -> {res['final_loss_jax']:.6f}")
    print(f"Max Loss Trajectory Diff: {res['max_loss_diff']:.2e}")
    print(f"Max Parameter |W1| Diff : {res['max_w1_diff']:.2e}")


if __name__ == "__main__":
    main()
