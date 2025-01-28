import jax
import jax.numpy as jnp
from . import jaxutils
from . import ninjax as nj
from . import nets

f32 = jnp.float32


def set_mask(row_mask, row_indices):
    return row_mask.at[row_indices].set(1.0)


class MoEGate(nj.Module):
    """
    MoEGate can do either soft gating (a continuous mixture over all experts)
    or the original top_k gating (discrete and non-differentiable).

    Args:
      moe_dim: Dimensionality of the input features.
      n_routed_experts: Total number of experts.
      n_expert_groups: Number of “group experts” if you want grouping.
      top_k: Number of experts selected in hard gating mode.
      top_k_groups: Number of “group-limited” top groups selected.
      score_func: 'softmax' or 'sigmoid' to generate gating scores.
      route_scale: Scalar multiplier for final gating scores.
      use_bias: Whether to add a bias to gating logits (only if moe_dim >= 4096).
      soft_gating: If True, skip top_k entirely and do a continuous mixture of experts.
    """

    def __init__(
        self,
        moe_dim,
        n_routed_experts,
        n_expert_groups=1,
        top_k=2,
        top_k_groups=1,
        score_func="softmax",
        route_scale=1.0,
        use_bias=False,
        soft_gating=False,
    ):
        self._moe_dim = moe_dim
        self._n_routed_experts = n_routed_experts
        self._n_expert_groups = n_expert_groups
        self._top_k = top_k
        self._top_k_groups = top_k_groups
        self._score_func = score_func
        self._route_scale = route_scale
        self._use_bias = use_bias and moe_dim >= 4096
        self._soft_gating = soft_gating

    def __call__(self, x):
        """
        Returns:
          weights: either shape [B, n_routed_experts] in soft-gating mode,
                   or shape [B, top_k] in hard-gating mode
          indices: either None in soft-gating mode,
                   or shape [B, top_k] (the chosen experts) in hard-gating mode
        """
        # Compute gating logits
        weight = self.get("weight", jnp.zeros, (self._n_routed_experts, self._moe_dim))
        logits = jnp.dot(x, weight.T)

        if self._use_bias:
            bias = self.get("bias", jnp.zeros, (self._n_routed_experts,))
            logits = logits + bias

        if self._score_func == "softmax":
            scores = jax.nn.softmax(logits.astype(f32), axis=-1)
        else:
            scores = jax.nn.sigmoid(logits)

        if self._n_expert_groups > 1:
            scores_g = scores.reshape(x.shape[0], self._n_expert_groups, -1)
            if self._use_bias:
                top_2_scores = jax.lax.top_k(scores_g, 2)[0]
                group_scores = jnp.sum(top_2_scores, axis=-1)
            else:
                group_scores = jnp.max(scores_g, axis=-1)
            indices_g = jax.lax.top_k(group_scores, self._top_k_groups)[1]
            mask = jax.vmap(set_mask)(jnp.zeros_like(scores_g[..., 0]), indices_g)
            scores_g = (scores_g * mask[..., None]).reshape(x.shape[0], -1)
            scores = scores_g

        if self._soft_gating:
            if self._score_func == "sigmoid":
                row_sum = jnp.sum(scores, axis=-1, keepdims=True) + 1e-8
                scores = scores / row_sum

            weights = scores * self._route_scale
            indices = None
            return weights, indices

        top_scores, indices = jax.lax.top_k(scores, self._top_k)

        if self._score_func == "sigmoid":
            top_scores = top_scores / (jnp.sum(top_scores, axis=-1, keepdims=True) + 1e-8)

        top_scores = top_scores * self._route_scale
        return top_scores, indices


class MoEExpert(nj.Module):
    """
    Single expert subnetwork.

    kw can contain act, norm, bias, outscale, outnorm, winit, fan, etc.
    """

    def __init__(self, moe_dim, inter_dim, symlog_inputs=False, **kw):
        self._moe_dim = moe_dim
        self._inter_dim = inter_dim
        self._symlog_inputs = symlog_inputs
        self._linear_kw = {
            "act": kw.get("act", "none"),
            "norm": kw.get("norm", "none"),
            "bias": kw.get("bias", True),
            "outscale": kw.get("outscale", 1.0),
            "outnorm": kw.get("outnorm", False),
            "winit": kw.get("winit", "uniform"),
            "fan": kw.get("fan", "avg"),
        }

    def __call__(self, x):
        if self._symlog_inputs:
            x = jaxutils.symlog(x)

        w1 = self.get("w1", nets.Linear, self._inter_dim, **self._linear_kw)(x)
        w3 = self.get("w3", nets.Linear, self._inter_dim, **self._linear_kw)(x)
        out = self.get("w2", nets.Linear, self._moe_dim, **self._linear_kw)(jax.nn.silu(w1) * w3)
        return out


class MoE(nj.Module):
    """
    Full MoE layer that uses MoEGate to get gating weights, runs that many experts,
    and optionally has shared_expert(s).
    """

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
        soft_gating=False,
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

        print(f"[MoE] Created MoE with {n_routed_experts} experts and {n_shared_experts} shared experts")
        print(f"[MoE]  - gating: {'soft' if soft_gating else f'top_k={top_k}'}")
        print(f"[MoE]  - output dist config: {self._dist}")

        self._gate_config = dict(
            moe_dim=moe_dim,
            n_routed_experts=n_routed_experts,
            n_expert_groups=n_expert_groups,
            top_k=top_k,
            top_k_groups=top_k_groups,
            score_func=score_func,
            route_scale=route_scale,
            soft_gating=soft_gating,
        )

    def _out(self, name, shape, x):
        return self.get(f"dist_{name}", nets.Dist, shape, **self._dist)(x)

    def __call__(self, x):
        x = self._inputs(x)
        x = jaxutils.cast_to_compute(x)

        # Gating
        weights, indices = self.get("gate", MoEGate, **self._gate_config)(x)

        experts = []
        for i in range(self._n_routed_experts):
            expert = self.get(f"expert_{i}", MoEExpert, self._moe_dim, self._moe_inter_dim, **self._kw)
            experts.append(expert)

        if indices is None:
            all_outs = []
            for i, expert in enumerate(experts):
                out_i = expert(x).astype(f32)  # [B, D]
                w_i = weights[:, i, None]  # [B, 1]
                all_outs.append(out_i * w_i)
            y = jnp.sum(jnp.stack(all_outs, axis=0), axis=0)
        else:
            y = jnp.zeros_like(x, dtype=f32)

            for i in range(self._n_routed_experts):
                mask = jnp.any(indices == i, axis=-1)  # [B]
                masked_input = jnp.where(mask[..., None], x, 0.0)
                out = experts[i](masked_input).astype(f32)
                multiplier = jnp.sum(jnp.where(indices == i, weights, 0.0), axis=-1)
                y += out * multiplier[..., None]

        if self._n_shared_experts > 0:
            shared_expert = self.get("shared_expert", MoEExpert, self._moe_dim, self._moe_inter_dim * self._n_shared_experts, **self._kw)
            y += shared_expert(x)

        if self._shape is None:
            return y
        elif isinstance(self._shape, tuple):
            return self._out("out", self._shape, y)
        elif isinstance(self._shape, dict):
            return {k: self._out(k, v, y) for k, v in self._shape.items()}
        else:
            raise ValueError(f"Unrecognized shape spec {self._shape}")
