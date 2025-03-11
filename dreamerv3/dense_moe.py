import jax
from . import nets, ninjax as nj


class MoE(nj.Module):
    """
    Mixture of Experts with dense routing
    """

    def __init__(
        self,
        shape=None,
        layers=5,
        units=1024,
        dims=None,
        inputs=["tensor"],
        moe_dim=1024,
        n_routed_experts=2,
        n_shared_experts=1,
        score_func="softmax",
        route_scale=1.0,
        **kw,
    ):
        # Save relevant configs
        self.shape = shape
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.score_func = score_func
        self.route_scale = route_scale
        self.dims = dims

        distkeys = ("dist", "outscale", "minstd", "maxstd", "outnorm", "unimix", "bins")

        print(f"inputs={inputs}, dims={dims}")
        self._inputs = nets.Input(inputs, dims=dims)

        # Create gate
        self.gate = nets.MLP(
            None, layers=2, units=moe_dim, dist="none", name="gate", **{k: v for k, v in kw.items() if k in distkeys and k != "dist"}
        )

        # Create experts
        self.experts = [
            nets.MLP(
                None,
                layers=layers,
                units=units,
                dist="none",
                name=f"expert_{expert_idx}",
                **{k: v for k, v in kw.items() if k in distkeys and k != "dist"},
            )
            for expert_idx in range(n_routed_experts)
        ]

        # Create shared expert
        if n_shared_experts > 0:
            self.shared_expert = nets.MLP(
                None, layers=layers, units=units, dist="none", name="shared_expert", **{k: v for k, v in kw.items() if k in distkeys and k != "dist"}
            )
        else:
            self.shared_expert = None

        # Convert mixture of logits to distributions
        if shape is not None:
            self.output_head = nets.MLP(shape, layers=1, units=units, name="moe_output", **{k: v for k, v in kw.items() if k in distkeys})

    def __call__(self, x):
        # Get gate weights
        x = self._inputs(x)
        gate_logits = self.gate(x)

        # Apply softmax to get routing weights
        if self.score_func == "softmax":
            weights = jax.nn.softmax(gate_logits, axis=-1)
        elif self.score_func == "sigmoid":
            weights = jax.nn.sigmoid(gate_logits)
            weights = weights / weights.sum(axis=-1, keepdims=True)
        else:
            raise ValueError(f"Unknown score function: {self.score_func}")

        # Scale weights if needed
        weights = weights * self.route_scale

        # Apply each expert and weight the outputs
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(x)
            # Reshape weights to broadcast properly
            expert_weight = weights[..., i : i + 1]
            weighted_output = expert_output * expert_weight
            expert_outputs.append(weighted_output)

        # Output from all routed experts
        mixture_output = sum(expert_outputs)

        # Add shared expert if it exists
        if self.shared_expert is not None:
            shared_output = self.shared_expert(x)
            mixture_output = mixture_output + shared_output

        if self.shape is not None:
            return self.output_head(mixture_output)
        else:
            return mixture_output
