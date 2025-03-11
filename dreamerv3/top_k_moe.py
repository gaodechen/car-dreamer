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
    MoEGate with top_k gating.

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
    ):
        self._moe_dim = moe_dim
        self._n_routed_experts = n_routed_experts
        self._n_expert_groups = n_expert_groups
        self._top_k = top_k
        self._top_k_groups = top_k_groups
        self._score_func = score_func
        self._route_scale = route_scale
        self._use_bias = use_bias and moe_dim >= 4096

    def __call__(self, x):
        """
        x: shape [B*T, D]

        Returns:
          weights: [B*T, top_k] in hard-gating mode
          indices: shape [B*T, top_k] (the chosen experts) in hard-gating mode
        """
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
            B_times_T = scores.shape[0]
            scores_g = scores.reshape(B_times_T, self._n_expert_groups, -1)
            if self._use_bias:
                top_2_scores = jax.lax.top_k(scores_g, 2)[0]  # shape [B*T, n_groups, 2]
                group_scores = jnp.sum(top_2_scores, axis=-1)  # [B*T, n_groups]
            else:
                group_scores = jnp.max(scores_g, axis=-1)  # [B*T, n_groups]
            indices_g = jax.lax.top_k(group_scores, self._top_k_groups)[1]  # [B*T, top_k_groups]
            mask = jax.vmap(set_mask)(jnp.zeros_like(group_scores), indices_g)
            # shape for mask is [B*T, n_groups], expand last dim for experts
            scores_g = (scores_g * mask[..., None]).reshape(B_times_T, -1)
            scores = scores_g

        top_scores, indices = jax.lax.top_k(scores, self._top_k)
        if self._score_func == "sigmoid":
            norm = jnp.sum(top_scores, axis=-1, keepdims=True) + 1e-8
            top_scores = top_scores / norm

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
        # x: shape [B*T, D]
        if self._symlog_inputs:
            x = jaxutils.symlog(x)

        w1 = self.get("w1", nets.Linear, self._inter_dim, **self._linear_kw)(x)
        w3 = self.get("w3", nets.Linear, self._inter_dim, **self._linear_kw)(x)
        out = self.get("w2", nets.Linear, self._moe_dim, **self._linear_kw)(jax.nn.silu(w1) * w3)
        return out  # shape [B*T, moe_dim]


class MoE(nj.Module):
    """
    Full MoE layer that uses MoEGate to get gating weights, runs that many experts,
    and optionally has shared_expert(s).

    Flatten [B, T, D] => [B*T, D]
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
        print(f"[MoE]  - output dist config: {self._dist}")

        self._gate_config = dict(
            moe_dim=moe_dim,
            n_routed_experts=n_routed_experts,
            n_expert_groups=n_expert_groups,
            top_k=top_k,
            top_k_groups=top_k_groups,
            score_func=score_func,
            route_scale=route_scale,
        )

    def _out(self, name, shape, x):
        return self.get(f"dist_{name}", nets.Dist, shape, **self._dist)(x)

    def __call__(self, x):
        """
        x can be shape [B, T, D] or just [B, D]. We'll flatten the leading dims
        into one if needed so gating has shape [B*T, D].
        """
        x = self._inputs(x)  # e.g. shape [B, T, D]
        x = jaxutils.cast_to_compute(x)

        # Flatten all but last dimension:
        leading_shape = x.shape[:-1]  # e.g. (B, T)
        d = x.shape[-1]  # D
        x2 = x.reshape((-1, d))  # [B*T, D]

        # Gating
        weights, indices = self.get("gate", MoEGate, **self._gate_config)(x2)
        # weights shape: [B*T, top_k]
        # indices shape: [B*T, top_k]

        # Build/fetch experts
        experts = []
        for i in range(self._n_routed_experts):
            expert = self.get(f"expert_{i}", MoEExpert, self._moe_dim, self._moe_inter_dim, **self._kw)
            experts.append(expert)

        # Hard gating => shape [B*T, top_k], indices [B*T, top_k]
        y2 = jnp.zeros_like(x2, dtype=f32)

        for i in range(self._n_routed_experts):
            # mask shape [B*T], True where this example picks expert i
            mask = jnp.any(indices == i, axis=-1)
            masked_input = jnp.where(mask[..., None], x2, 0.0)
            out_i = experts[i](masked_input).astype(f32)
            # gather gating weights for expert i
            multiplier = jnp.sum(jnp.where(indices == i, weights, 0.0), axis=-1)
            # broadcast that multiplier
            y2 += out_i * multiplier[..., None]

        # Optionally add shared experts
        if self._n_shared_experts > 0:
            shared_expert = self.get("shared_expert", MoEExpert, self._moe_dim, self._moe_inter_dim * self._n_shared_experts, **self._kw)
            # shape [B*T, D]
            y2 += shared_expert(x2)

        # Reshape back to [B, T, D] or [B, D]
        y = y2.reshape(leading_shape + (self._moe_dim,))

        # Possibly pass to a distribution head
        if self._shape is None:
            return y
        elif isinstance(self._shape, tuple):
            return self._out("out", self._shape, y)
        elif isinstance(self._shape, dict):
            return {k: self._out(k, v, y) for k, v in self._shape.items()}
        else:
            raise ValueError(f"Unrecognized shape spec {self._shape}")
