
"""
Uses MAML utils to meta-learn a deep linear network for binary classification.
"""

import numpy as np
import jax
import jax.numpy as jnp
import flax
import optax

from metax import maml

SHOTS = 16
FEATURES = 128
ADAPT_STEPS = 3
ITERATIONS = 100


class LinearMLP(flax.linen.Module):

    sizes = [32, 1]

    @flax.linen.compact
    def __call__(self, x):
        for size in self.sizes:
            x = flax.linen.Dense(size)(x)
        return x


def mse_loss(weights, model, X, y):
    preds = model.apply(weights, X)
    return jnp.square(preds - y).mean()


def main():

    np.random.seed(1234)
    rng = jax.random.PRNGKey(1234)

    # intialize model and optimizer
    model = LinearMLP()
    params = model.init(rng, jnp.ones((SHOTS, FEATURES)))
    optimizer = optax.adam(3e-3)
    optimizer_state = optimizer.init(params)

    # define meta-loss with 3 Adam steps
    fast_adaptation = maml.OptaxAdaptation(
        optimizer=optax.adam(0.001, eps=1e-8),
        steps=ADAPT_STEPS,
    )

    def meta_loss(weights, X_adapt, y_adapt, X_eval, y_eval):
        adapted_weights = fast_adaptation(
            weights,
            mse_loss,
            model,
            X_adapt,
            y_adapt,
        )
        eval_loss = mse_loss(adapted_weights, model, X_eval, y_eval)
        return eval_loss, eval_loss
    meta_grads = jax.jit(jax.grad(meta_loss, has_aux=True))

    for iteration in range(ITERATIONS):

        # sample fake data
        ground_truth = flax.linen.Dense(1)
        gt_params = ground_truth.init(rng, jnp.ones((SHOTS, FEATURES)))
        X_adapt = jnp.array(np.random.randn(SHOTS, FEATURES))
        X_eval = jnp.array(np.random.randn(SHOTS, FEATURES))
        y_adapt = ground_truth.apply(gt_params, X_adapt)
        y_eval = ground_truth.apply(gt_params, X_eval)

        # update initialization
        grads, loss = meta_grads(params, X_adapt, y_adapt, X_eval, y_eval)
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        params = optax.apply_updates(params, updates)
        print(f'{iteration}: {loss:.4f}')


if __name__ == "__main__":
    main()
