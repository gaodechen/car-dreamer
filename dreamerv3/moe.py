import jax
import jax.numpy as jnp
from . import jaxutils
from . import ninjax as nj
from . import nets

f32 = jnp.float32


def set_mask(row_mask, row_indices):
    return row_mask.at[row_indices].set(1.0)


class MoEGate(nj.Module):
    def __init__(self, moe_dim, n_routed_experts, n_expert_groups=1, top_k=2, top_k_groups=1, score_func="softmax", route_scale=1.0, use_bias=False):
        self._moe_dim = moe_dim
        self._n_routed_experts = n_routed_experts
        self._n_expert_groups = n_expert_groups
        self._top_k = top_k
        self._top_k_groups = top_k_groups
        self._score_func = score_func
        self._route_scale = route_scale
        self._use_bias = use_bias and moe_dim >= 4096

    def __call__(self, x):
        weight = self.get("weight", jnp.zeros, (self._n_routed_experts, self._moe_dim))
        scores = jnp.dot(x, weight.T)

        if self._score_func == "softmax":
            scores = jax.nn.softmax(scores.astype(f32), axis=-1)
        else:
            scores = jax.nn.sigmoid(scores)

        original_scores = scores

        if self._use_bias:
            bias = self.get("bias", jnp.zeros, (self._n_routed_experts,))
            scores = scores + bias

        if self._n_expert_groups > 1:
            scores = scores.reshape(x.shape[0], self._n_expert_groups, -1)
            if not self._use_bias:
                group_scores = jnp.max(scores, axis=-1)
            else:
                top_2_scores = jax.lax.top_k(scores, 2)[0]
                group_scores = jnp.sum(top_2_scores, axis=-1)
            indices = jax.lax.top_k(group_scores, self._top_k_groups)[1]
            mask = jax.vmap(set_mask)(jnp.zeros_like(scores[..., 0]), indices)
            scores = (scores * mask[..., None]).reshape(x.shape[0], -1)

        _, indices = jax.lax.top_k(scores, self._top_k)
        weights = jnp.take_along_axis(original_scores, indices, axis=-1)

        if self._score_func == "sigmoid":
            weights = weights / jnp.sum(weights, axis=-1, keepdims=True)

        return weights * self._route_scale, indices


class MoEExpert(nj.Module):
    def __init__(self, moe_dim, inter_dim, symlog_inputs=False, **kw):
        self._moe_dim = moe_dim
        self._inter_dim = inter_dim
        self._symlog_inputs = symlog_inputs
        self._linear_kw = {
            "act": kw["act"] if "act" in kw else "none",
            "norm": kw["norm"] if "norm" in kw else "none",
            "bias": kw["bias"] if "bias" in kw else True,
            "outscale": kw["outscale"] if "outscale" in kw else 1.0,
            "outnorm": kw["outnorm"] if "outnorm" in kw else False,
            "winit": kw["winit"] if "winit" in kw else "uniform",
            "fan": kw["fan"] if "fan" in kw else "avg",
        }

    def __call__(self, x):
        if self._symlog_inputs:
            x = jaxutils.symlog(x)

        w1 = self.get("w1", nets.Linear, self._inter_dim, **self._linear_kw)(x)
        w3 = self.get("w3", nets.Linear, self._inter_dim, **self._linear_kw)(x)
        return self.get("w2", nets.Linear, self._moe_dim, **self._linear_kw)(jax.nn.silu(w1) * w3)


class MoE(nj.Module):
    def __init__(
        self,
        shape=None,
        moe_dim=None,
        moe_inter_dim=None,
        n_routed_experts=8,
        n_shared_experts=1,
        n_expert_groups=1,
        top_k=2,
        top_k_groups=1,
        score_func="softmax",
        route_scale=1.0,
        inputs=["tensor"],
        dims=None,
        **kw,
    ):
        self._shape = shape if isinstance(shape, (tuple, list)) else (shape,)
        self._moe_dim = moe_dim
        self._moe_inter_dim = moe_inter_dim
        self._n_routed_experts = n_routed_experts
        self._n_shared_experts = n_shared_experts
        self._inputs = nets.Input(inputs, dims=dims)
        self._kw = kw

        distkeys = ("dist", "outscale", "minstd", "maxstd", "outnorm", "unimix", "bins")
        self._dist = {k: v for k, v in kw.items() if k in distkeys}

        self._gate_config = {
            "moe_dim": moe_dim,
            "n_routed_experts": n_routed_experts,
            "n_expert_groups": n_expert_groups,
            "top_k": top_k,
            "top_k_groups": top_k_groups,
            "score_func": score_func,
            "route_scale": route_scale,
        }

    def _out(self, name, shape, x):
        return self.get(f"dist_{name}", nets.Dist, shape, **self._dist)(x)

    def __call__(self, x):
        x = self._inputs(x)
        x = jaxutils.cast_to_compute(x)

        weights, indices = self.get("gate", MoEGate, **self._gate_config)(x)

        counts = jnp.zeros((self._n_routed_experts,), dtype=jnp.int32)
        for i in range(self._n_routed_experts):
            counts = counts.at[i].set(jnp.sum(indices == i))

        experts = []
        for i in range(self._n_routed_experts):
            expert = self.get(f"expert_{i}", MoEExpert, self._moe_dim, self._moe_inter_dim, **self._kw)
            experts.append(expert)

        y = jnp.zeros_like(x, dtype=f32)

        for i, expert in enumerate(experts):
            mask = jnp.any(indices == i, axis=-1)

            masked_input = jnp.where(mask[..., None], x, 0)
            out = expert(masked_input).astype(f32)
            multiplier = jnp.sum(jnp.where(indices == i, weights, 0), axis=-1)

            y += out * multiplier[..., None]

        if self._n_shared_experts > 0:
            shared_expert = self.get("shared_expert", MoEExpert, self._moe_dim, self._moe_inter_dim * self._n_shared_experts, **self._kw)
            y = y + shared_expert(x)

        if self._shape is None:
            return y
        elif isinstance(self._shape, tuple):
            return self._out("out", self._shape, y)
        elif isinstance(self._shape, dict):
            return {k: self._out(k, v, y) for k, v in self._shape.items()}
        else:
            raise ValueError(self._shape)
