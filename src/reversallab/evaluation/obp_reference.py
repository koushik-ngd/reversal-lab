"""Point estimates from Open Bandit Pipeline's IPW and SNIPW classes.

OBP 0.5.5 declares ``python >=3.7.1,<3.10`` and imports PyTorch, pandas, SciPy,
and scikit-learn from its package ``__init__``. This process is Python 3.12.
Importing that stack also failed here: Windows Application Control refused to
load SciPy's compiled extension. The point-estimate methods do not call those
libraries. This module therefore loads ``InverseProbabilityWeighting`` and
``SelfNormalizedInverseProbabilityWeighting`` from a local checkout of
``st-tech/zr-obp`` in a subprocess, after substituting empty modules for the
unused imports. The arithmetic and ``check_ope_inputs`` are the upstream
functions. The checkout is not downloaded by this module.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from reversallab.exceptions import ConfigurationError

_BRIDGE = textwrap.dedent(
    """\
    import importlib
    import json
    import sys
    import types
    from pathlib import Path

    import numpy as np

    def _check_scalar(
        value, name, target_type, min_val=None, max_val=None, include_boundaries="both"
    ):
        if isinstance(target_type, tuple):
            numeric = isinstance(value, tuple((*target_type, np.integer, np.floating)))
        else:
            numeric = isinstance(value, (target_type, np.integer, np.floating))
        if not numeric:
            raise TypeError(f"{name} has type {type(value)}")
        if min_val is not None and value < min_val:
            raise ValueError(f"{name} is below {min_val}")
        if max_val is not None and value > max_val:
            raise ValueError(f"{name} is above {max_val}")
        return value

    sklearn_utils = types.ModuleType("sklearn.utils")
    sklearn_utils.check_scalar = _check_scalar
    sklearn_utils.check_random_state = lambda seed=None: np.random.RandomState(0)
    sklearn = types.ModuleType("sklearn")
    sklearn.utils = sklearn_utils
    sys.modules["sklearn"] = sklearn
    sys.modules["sklearn.utils"] = sklearn_utils

    torch = types.ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})
    # obp.utils annotates check_tensor with torch.tensor, evaluated at import.
    torch.tensor = torch.Tensor
    sys.modules["torch"] = torch
    sys.modules["pandas"] = types.ModuleType("pandas")
    scipy = types.ModuleType("scipy")
    stats = types.ModuleType("scipy.stats")
    scipy.stats = stats
    sys.modules["scipy"] = scipy
    sys.modules["scipy.stats"] = stats

    root = Path(sys.argv[1])
    obp = types.ModuleType("obp")
    obp.__path__ = [str(root / "obp")]
    obp.__package__ = "obp"
    sys.modules["obp"] = obp
    ope = types.ModuleType("obp.ope")
    ope.__path__ = [str(root / "obp" / "ope")]
    ope.__package__ = "obp.ope"
    sys.modules["obp.ope"] = ope
    importlib.import_module("obp.utils")
    estimators = importlib.import_module("obp.ope.estimators")

    payloads = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    ipw = estimators.InverseProbabilityWeighting()
    snipw = estimators.SelfNormalizedInverseProbabilityWeighting()
    source = Path(estimators.__file__)
    results = []
    for payload in payloads:
        reward = np.asarray(payload["reward"], dtype=np.float64)
        action = np.asarray(payload["action"], dtype=np.int64)
        pscore = np.asarray(payload["pscore"], dtype=np.float64)
        action_dist = np.asarray(payload["action_dist"], dtype=np.float64)[:, :, None]
        position = np.zeros(reward.shape[0], dtype=np.int64)
        results.append({
            "ipw": float(ipw.estimate_policy_value(
                reward=reward, action=action, action_dist=action_dist,
                pscore=pscore, position=position,
            )),
            "snipw": float(snipw.estimate_policy_value(
                reward=reward, action=action, action_dist=action_dist,
                pscore=pscore, position=position,
            )),
        })
    Path(sys.argv[3]).write_text(
        json.dumps({
            "lambda": float(ipw.lambda_),
            "estimators_file": str(source),
            "estimates": results,
        }),
        encoding="utf-8",
    )
    """
)


def obp_estimators_path(obp_root: Path) -> Path:
    """Return the IPW/SNIPW source file inside a zr-obp checkout."""
    path = obp_root / "obp" / "ope" / "estimators.py"
    if not path.is_file():
        raise ConfigurationError(
            f"Open Bandit Pipeline source was not found at {path}. "
            "Clone https://github.com/st-tech/zr-obp (the library and its small `obd/` "
            "sample, not the full 26M-row dataset) so this path exists. "
            "Example: git clone --depth 1 --filter=blob:none --sparse "
            "https://github.com/st-tech/zr-obp.git data/raw/zr-obp "
            "&& git -C data/raw/zr-obp sparse-checkout set obd obp"
        )
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def obp_point_estimates(
    obp_root: Path,
    *,
    reward: npt.NDArray[np.float64],
    action: npt.NDArray[np.int64],
    pscore: npt.NDArray[np.float64],
    action_dist: npt.NDArray[np.float64],
) -> dict[str, Any]:
    """Run OBP's IPW and SNIPW on one set of arrays. See :func:`obp_point_estimates_many`."""
    batch = obp_point_estimates_many(
        obp_root,
        reward=[reward],
        action=[action],
        pscore=[pscore],
        action_dist=[action_dist],
    )
    estimate = batch["estimates"][0]
    return {
        "ipw": estimate["ipw"],
        "snipw": estimate["snipw"],
        "lambda": batch["lambda"],
        "estimators_sha256": batch["estimators_sha256"],
    }


def obp_point_estimates_many(
    obp_root: Path,
    *,
    reward: list[npt.NDArray[np.float64]],
    action: list[npt.NDArray[np.int64]],
    pscore: list[npt.NDArray[np.float64]],
    action_dist: list[npt.NDArray[np.float64]],
) -> dict[str, Any]:
    """Run OBP's IPW and SNIPW on several array sets in one process.

    Each ``action_dist`` has shape ``(N, K)``. OBP stores evaluation probabilities
    as ``(N, K, len_list)``. Each of our rows is already one position, so
    ``len_list`` is 1 and OBP's position index is 0. That index is not the
    ZOZOTOWN position stored in the file.
    """
    if not reward or len({len(reward), len(action), len(pscore), len(action_dist)}) != 1:
        raise ConfigurationError("OBP comparison cases must be a non-empty aligned list.")
    source = obp_estimators_path(obp_root)
    payload = [
        {
            "reward": np.asarray(one_reward, dtype=np.float64).tolist(),
            "action": np.asarray(one_action, dtype=np.int64).tolist(),
            "pscore": np.asarray(one_pscore, dtype=np.float64).tolist(),
            "action_dist": np.asarray(one_dist, dtype=np.float64).tolist(),
        }
        for one_reward, one_action, one_pscore, one_dist in zip(
            reward, action, pscore, action_dist, strict=True
        )
    ]
    from tempfile import TemporaryDirectory

    with TemporaryDirectory(prefix="reversallab-obp-") as tmp:
        directory = Path(tmp)
        request = directory / "request.json"
        response = directory / "response.json"
        request.write_text(json.dumps(payload), encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, "-c", _BRIDGE, str(obp_root), str(request), str(response)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0 or not response.is_file():
            detail = (completed.stderr or completed.stdout or "no output").strip()
            raise ConfigurationError(
                f"Open Bandit Pipeline IPW/SNIPW failed to run from {source}.\n{detail}"
            )
        parsed: dict[str, Any] = json.loads(response.read_text(encoding="utf-8"))
    parsed["estimators_sha256"] = _file_sha256(source)
    if parsed["lambda"] != float("inf"):
        raise ConfigurationError(
            f"OBP IPW lambda_ is {parsed['lambda']}, so weights would be clipped. "
            "ReversalLab does not clip. Refusing to treat the two estimates as the same method."
        )
    if len(parsed["estimates"]) != len(payload):
        raise ConfigurationError(
            "Open Bandit Pipeline returned "
            f"{len(parsed['estimates'])} estimates for {len(payload)} cases."
        )
    return parsed
