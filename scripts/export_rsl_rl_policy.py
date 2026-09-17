#!/usr/bin/env python3
"""Export an RSL-RL actor checkpoint to TorchScript and ONNX for deployment.

RSL-RL writes ``model_<iter>.pt`` files that hold the full actor-critic state
and, when ``empirical_normalization`` is enabled, the running observation
statistics.  The deployment policies expect a single callable that maps a raw
observation to the mean action, so this script rebuilds the actor MLP, folds
the observation normalizer in front of it, and writes:

    <output>.pt    TorchScript module (loaded by ``TorchScriptRunner``)
    <output>.onnx  ONNX graph (loaded by ``CpuOnnxRunner``)

The MLP layout is inferred from the ``actor.*`` weight shapes.  The activation
and the normalization flag are read from ``params/agent.yaml`` next to the
checkpoint when it exists; otherwise ELU is assumed and the normalizer is used
whenever the checkpoint contains one.

Example:

    python scripts/export_rsl_rl_policy.py \
        --checkpoint logs/rsl_rl/k1_flat/<run>/model_9999.pt \
        --output tasks/bm154/robots/k1/models/k1_dance_jamesbrown_marg_bm154
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import torch
import torch.nn as nn

_ACTIVATIONS = {
    "elu": nn.ELU,
    "selu": nn.SELU,
    "relu": nn.ReLU,
    "crelu": nn.CELU,
    "lrelu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "identity": nn.Identity,
}


class ExportedPolicy(nn.Module):
    """Observation normalization followed by the actor MLP."""

    def __init__(
        self,
        actor: nn.Sequential,
        obs_mean: torch.Tensor,
        obs_std: torch.Tensor,
        eps: float,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.register_buffer("obs_mean", obs_mean)
        self.register_buffer("obs_std", obs_std)
        self.eps = float(eps)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor((obs - self.obs_mean) / (self.obs_std + self.eps))


def _load_agent_cfg(checkpoint: str) -> dict:
    path = os.path.join(os.path.dirname(checkpoint), "params", "agent.yaml")
    if not os.path.isfile(path):
        return {}
    try:
        import yaml
    except ImportError:
        print(f"[warn] pyyaml not installed; ignoring {path}")
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_actor(state_dict: dict, activation: str) -> nn.Sequential:
    """Rebuild the RSL-RL actor MLP from ``actor.<i>.weight`` shapes."""
    layer_ids = sorted(
        int(m.group(1))
        for key in state_dict
        for m in [re.match(r"^actor\.(\d+)\.weight$", key)]
        if m is not None
    )
    if not layer_ids:
        raise ValueError("checkpoint has no 'actor.*.weight' entries")
    if activation not in _ACTIVATIONS:
        raise ValueError(
            f"unknown activation '{activation}', expected one of "
            f"{sorted(_ACTIVATIONS)}")

    layers: list[nn.Module] = []
    for n, idx in enumerate(layer_ids):
        weight = state_dict[f"actor.{idx}.weight"]
        linear = nn.Linear(weight.shape[1], weight.shape[0])
        linear.weight.data.copy_(weight)
        linear.bias.data.copy_(state_dict[f"actor.{idx}.bias"])
        layers.append(linear)
        if n < len(layer_ids) - 1:
            layers.append(_ACTIVATIONS[activation]())
    return nn.Sequential(*layers)


def build_policy(
    checkpoint: str,
    activation: str | None = None,
    use_normalizer: bool | None = None,
    normalizer_eps: float = 1e-2,
) -> tuple[ExportedPolicy, dict]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    agent_cfg = _load_agent_cfg(checkpoint)
    policy_cfg = agent_cfg.get("policy", {})

    if activation is None:
        activation = str(policy_cfg.get("activation", "elu"))
    actor = build_actor(state_dict, activation)
    obs_dim = actor[0].in_features
    act_dim = actor[-1].out_features

    norm_state = ckpt.get("obs_norm_state_dict")
    if use_normalizer is None:
        use_normalizer = bool(
            agent_cfg.get("empirical_normalization", norm_state is not None))
    if use_normalizer:
        if norm_state is None:
            raise ValueError(
                "empirical normalization requested but the checkpoint has no "
                "'obs_norm_state_dict'")
        obs_mean = norm_state["_mean"].reshape(-1).clone().float()
        obs_std = norm_state["_std"].reshape(-1).clone().float()
        if obs_mean.numel() != obs_dim:
            raise ValueError(
                f"normalizer size {obs_mean.numel()} does not match actor "
                f"input size {obs_dim}")
        eps = normalizer_eps
    else:
        obs_mean = torch.zeros(obs_dim)
        obs_std = torch.ones(obs_dim)
        eps = 0.0

    policy = ExportedPolicy(actor, obs_mean, obs_std, eps).eval()
    info = {
        "checkpoint": os.path.abspath(checkpoint),
        "iteration": ckpt.get("iter"),
        "num_observations": obs_dim,
        "num_actions": act_dim,
        "hidden_dims": [layer.out_features for layer in actor[:-1]
                        if isinstance(layer, nn.Linear)],
        "activation": activation,
        "normalizer": use_normalizer,
        "normalizer_eps": eps,
    }
    return policy, info


def export_torchscript(policy: ExportedPolicy, path: str) -> None:
    scripted = torch.jit.script(policy)
    scripted.save(path)


def export_onnx(
    policy: ExportedPolicy, path: str, opset: int, info: dict,
) -> None:
    obs = torch.zeros(1, info["num_observations"])
    torch.onnx.export(
        policy,
        (obs,),
        path,
        export_params=True,
        opset_version=opset,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
    )
    try:
        import onnx
    except ImportError:
        return
    model = onnx.load(path)
    for key, value in info.items():
        entry = onnx.StringStringEntryProto()
        entry.key = key
        entry.value = str(value)
        model.metadata_props.append(entry)
    onnx.save(model, path)


def verify(policy: ExportedPolicy, outputs: dict[str, str], info: dict) -> None:
    torch.manual_seed(0)
    obs = torch.randn(8, info["num_observations"]) * 2.0
    with torch.no_grad():
        reference = policy(obs)
    if "pt" in outputs:
        scripted = torch.jit.load(outputs["pt"], map_location="cpu")
        with torch.no_grad():
            diff = (scripted(obs) - reference).abs().max().item()
        print(f"[verify] torchscript max |diff| = {diff:.3e}")
    if "onnx" in outputs:
        try:
            import onnxruntime as ort
        except ImportError:
            print("[verify] onnxruntime not installed; skipping ONNX check")
            return
        session = ort.InferenceSession(
            outputs["onnx"], providers=["CPUExecutionProvider"])
        diff = 0.0
        for i in range(obs.shape[0]):
            out = session.run(["actions"], {"obs": obs[i:i + 1].numpy()})[0]
            diff = max(diff, float(
                (torch.from_numpy(out) - reference[i:i + 1]).abs().max()))
        print(f"[verify] onnx max |diff| = {diff:.3e}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True,
                        help="RSL-RL model_<iter>.pt checkpoint")
    parser.add_argument("--output", required=True,
                        help="output path prefix (extension is appended)")
    parser.add_argument("--formats", nargs="+", choices=("pt", "onnx"),
                        default=("pt", "onnx"), help="formats to write")
    parser.add_argument(
        "--activation", default=None,
        help="actor activation (default: params/agent.yaml or elu)")
    norm = parser.add_mutually_exclusive_group()
    norm.add_argument(
        "--normalizer", dest="normalizer", action="store_true", default=None,
        help="force folding the observation normalizer")
    norm.add_argument(
        "--no-normalizer", dest="normalizer", action="store_false",
        help="export the raw actor without observation normalization")
    parser.add_argument(
        "--normalizer-eps", type=float, default=1e-2,
        help="epsilon of rsl_rl EmpiricalNormalization (default 1e-2)")
    parser.add_argument("--opset", type=int, default=11,
                        help="ONNX opset version")
    args = parser.parse_args(argv)

    policy, info = build_policy(
        args.checkpoint,
        activation=args.activation,
        use_normalizer=args.normalizer,
        normalizer_eps=args.normalizer_eps,
    )
    for key, value in info.items():
        print(f"[info] {key}: {value}")

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    outputs: dict[str, str] = {}
    if "pt" in args.formats:
        outputs["pt"] = args.output + ".pt"
        export_torchscript(policy, outputs["pt"])
        print(f"[export] wrote {outputs['pt']}")
    if "onnx" in args.formats:
        outputs["onnx"] = args.output + ".onnx"
        export_onnx(policy, outputs["onnx"], args.opset, info)
        print(f"[export] wrote {outputs['onnx']}")
    verify(policy, outputs, info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
