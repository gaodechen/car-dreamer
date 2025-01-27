import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from . import jaxutils
from . import ninjax as nj
from . import nets

tfd = tfp.distributions
f32 = jnp.float32


class MoEGate(nj.Module):
    def __init__(self, dim, n_routed_experts, n_expert_groups=1, top_k=2, top_k_groups=1, score_func="softmax", route_scale=1.0, use_bias=False):
        self._dim = dim
        self._n_routed_experts = n_routed_experts
        self._n_expert_groups = n_expert_groups
        self._top_k = top_k
        self._top_k_groups = top_k_groups
        self._score_func = score_func
        self._route_scale = route_scale
        self._use_bias = use_bias and dim >= 4096

    def __call__(self, x):
        weight = self.get("weight", jnp.zeros, (self._n_routed_experts, self._dim))
        scores = jnp.dot(x, weight.T)  # Transpose to match dimensions

        if self._use_bias:
            bias = self.get("bias", jnp.zeros, (self._n_routed_experts,))
            scores = scores + bias

        if self._score_func == "softmax":
            scores = jax.nn.softmax(scores.astype(f32), axis=-1)
        else:  # sigmoid
            scores = jax.nn.sigmoid(scores)

        original_scores = scores

        if self._n_expert_groups > 1:
            scores = scores.reshape(x.shape[0], self._n_expert_groups, -1)
            if not self._use_bias:
                group_scores = jnp.max(scores, axis=-1)
            else:
                # Match DeepSeek's behavior for bias case
                top_2_scores = jax.lax.top_k(scores, 2)[0]
                group_scores = jnp.sum(top_2_scores, axis=-1)
            indices = jax.lax.top_k(group_scores, self._top_k_groups)[1]
            mask = jnp.zeros_like(scores[..., 0]).at[indices].set(1.0)
            scores = (scores * mask[..., None]).reshape(x.shape[0], -1)

        _, indices = jax.lax.top_k(scores, self._top_k)
        weights = jnp.take_along_axis(original_scores, indices, axis=-1)

        if self._score_func == "sigmoid":
            weights = weights / jnp.sum(weights, axis=-1, keepdims=True)

        return weights * self._route_scale, indices


class MoEExpert(nj.Module):
    def __init__(self, dim, inter_dim, symlog_inputs=False, **kw):
        self._dim = dim
        self._inter_dim = inter_dim
        self._symlog_inputs = symlog_inputs
        self._kw = kw

    def __call__(self, x):
        if self._symlog_inputs:
            x = jaxutils.symlog(x)

        w1 = self.get("w1", nets.Linear, self._inter_dim, **self._kw)(x)
        w3 = self.get("w3", nets.Linear, self._inter_dim, **self._kw)(x)
        return self.get("w2", nets.Linear, self._dim, **self._kw)(jax.nn.silu(w1) * w3)


class MoE(nj.Module):
    def __init__(
        self,
        shape=None,
        dim=None,
        n_routed_experts=8,
        n_shared_experts=1,
        n_expert_groups=1,
        top_k=2,
        top_k_groups=1,
        score_func="softmax",
        route_scale=1.0,
        symlog_inputs=False,
        **kw,
    ):
        self._shape = shape if isinstance(shape, (tuple, list)) else (shape,)
        self._dim = dim
        self._n_routed_experts = n_routed_experts
        self._n_shared_experts = n_shared_experts
        self._symlog_inputs = symlog_inputs
        self._kw = kw

        self._gate_config = {
            "dim": dim,
            "n_routed_experts": n_routed_experts,
            "n_expert_groups": n_expert_groups,
            "top_k": top_k,
            "top_k_groups": top_k_groups,
            "score_func": score_func,
            "route_scale": route_scale,
        }

    def __call__(self, x):
        if self._symlog_inputs:
            x = jaxutils.symlog(x)

        weights, indices = self.get("gate", MoEGate, **self._gate_config)(x)

        # Count expert usage for load balancing
        counts = jnp.zeros((self._n_routed_experts,), dtype=jnp.int32)
        for i in range(self._n_routed_experts):
            counts = counts.at[i].set(jnp.sum(indices == i))

        def process_expert(i, accumulator):
            mask = indices == i
            if not jnp.any(mask):
                return accumulator
            expert = self.get(f"expert_{i}", MoEExpert, self._dim, self._dim * 4, **self._kw)
            masked_input = jnp.where(mask[..., None], x, 0)
            out = expert(masked_input)
            return accumulator + out * weights[..., jnp.where(indices == i)[1], None]

        y = jax.lax.fori_loop(0, self._n_routed_experts, process_expert, jnp.zeros_like(x))

        # Shared experts (always active)
        if self._n_shared_experts > 0:
            shared_expert = self.get("shared_expert", MoEExpert, self._dim, self._dim * 4, **self._kw)
            y = y + shared_expert(x)

        if self._shape:
            return self.get("dist", nets.Dist, self._shape, **self._kw)(y)
        return y
