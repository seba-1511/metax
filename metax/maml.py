# -*- coding=utf-8 -*-

"""
Utilities to implement MAML-style meta-learning.
"""

import jax
import optax


class OptaxAdaptation:

    def __init__(self, optimizer, steps=1, with_memory=False):
        self.steps = steps
        self.optimizer = optimizer
        self.with_memory = with_memory
        self.optimizer_state = None

    def __call__(self, params, loss, *args, **kwargs):
        grad = jax.grad(loss)
        if self.with_memory and self.optimizer_state is not None:
            state = self.optimizer_state
        else:
            state = self.optimizer.init(params)
        for step in range(self.steps):
            gradients = grad(params, *args, **kwargs)
            updates, state = self.optimizer.update(gradients, state)
            params = optax.apply_updates(params, updates)
        if self.with_memory:
            self.optimizer_state = state
        return params


class SGD:

    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def __call__(self, params, grads):
        return params - self.learning_rate * grads


class GradientAdaptation:

    def __init__(self, update=None, steps=1):
        if update is None:
            update = SGD()
        self.update = update
        self.steps = steps

    def __call__(self, params, loss, *args, **kwargs):
        grad = jax.grad(loss)
        for step in range(self.steps):
            gradients = grad(params, *args, **kwargs)
            params = jax.tree_map(self.update, params, gradients)
        return params


class FastAdaptationLoss:

    """
    TODO: This doesn't support sampling different data
          for adaptation and evaluation.
    """

    def __init__(self, loss, adapt=None):
        if adapt is None:
            adapt = GradientAdaptation(SGD())
        self.adapt = adapt
        self.loss = loss

    def __call__(self, params, *args, **kwargs):
        adapted_params = self.adapt(params, self.loss, *args, **kwargs)
        return self.loss(adapted_params, *args, **kwargs)
