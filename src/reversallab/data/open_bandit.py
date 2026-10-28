"""Position-level adapter for the small Open Bandit Dataset sample.

The statistical meaning of the loaded fields is specified in
``docs/open_bandit_estimand.md``. This module reads the files. It does not
decide that a row is one slot of a three-item slate, and it does not impute
propensities, actions, or clicks.
"""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from reversallab.contracts.logged_data import LoggedBanditData
from reversallab.contracts.validation import FloatArray, IntArray
from reversallab.exceptions import ConfigurationError, DataValidationError

BEHAVIOR_POLICIES = ("random", "bts")
CAMPAIGNS = ("all", "men", "women")
RAW_POSITIONS = (1, 2, 3)
USER_FEATURE_COLUMNS = tuple(f"user_feature_{i}" for i in range(4))
PROBABILITY_COLUMN = "propensity_score"
# The dataset README names this column action_prob. The files shipped in zr-obp do not.


@dataclass(frozen=True)
class OpenBanditProvenance:
    """Identity of the file that was read, not an estimate."""

    behavior_policy: str
    campaign: str
    behavior_path: str
    behavior_sha256: str
    item_context_path: str
    item_context_sha256: str
    n_rows: int
    n_actions: int
    logging_distribution: str
    """How the logging distribution was obtained.

    ``uniform_identified``: every logged propensity equals ``1/K``. The adapter
    then stores a uniform ``logging_action_dist``. The file itself only records
    the displayed action's probability; the uniform matrix is the random policy
    that this constant column is consistent with. ``observed_action_only``: the
    other actions' probabilities are left missing, and support is not checked.
    """
    n_repeated_timestamps: int
    max_rows_per_timestamp: int
    n_user_feature_tuples: int


@dataclass(frozen=True)
class OpenBanditData:
    """One campaign log. Each row is one displayed item in one interface position.

    ``logged.context`` is the user-item affinity block, columns
    ``user-item_affinity_0`` through ``user-item_affinity_{K-1}``, in that order.
    It is not the one-hot user encoding produced by ``OpenBanditDataset.pre_process``.
    IPS and SNIPS do not read it. Candidate probabilities are passed separately.

    ``logged.group_id`` is the four user-feature hashes joined by ``|``. Equal
    ids mark repeated feature tuples. They are not impression ids, and the
    estimators do not resample by them.
    """

    logged: LoggedBanditData
    position: IntArray
    timestamp: tuple[str, ...]
    user_features: tuple[tuple[str, str, str, str], ...]
    item_feature_0: FloatArray
    """``item_feature_0`` from ``item_context.csv``, indexed by ``item_id``."""
    item_feature_labels: tuple[tuple[str, str, str], ...]
    """``(item_feature_1, item_feature_2, item_feature_3)`` indexed by ``item_id``."""
    provenance: OpenBanditProvenance

    @property
    def n_samples(self) -> int:
        return self.logged.n_samples

    @property
    def n_actions(self) -> int:
        return self.logged.n_actions

    def select_position(self, position: int) -> OpenBanditData:
        """Restrict the log to one raw ZOZOTOWN position in ``{1, 2, 3}``."""
        if position not in RAW_POSITIONS:
            raise ConfigurationError(
                f"ZOZOTOWN position must be one of {list(RAW_POSITIONS)}, got {position}."
            )
        mask = self.position == position
        if not np.any(mask):
            raise DataValidationError(f"No rows have position {position}.")
        # Provenance still describes the source file, including its original row count.
        return self._take(mask)

    def _take(self, mask: npt.NDArray[np.bool_]) -> OpenBanditData:
        idx = np.flatnonzero(mask)
        logged = self.logged
        timestamps = tuple(self.timestamp[i] for i in idx)
        users = tuple(self.user_features[i] for i in idx)
        return OpenBanditData(
            logged=LoggedBanditData(
                context=logged.context[idx],
                action=logged.action[idx],
                reward=logged.reward[idx],
                logging_propensity=logged.logging_propensity[idx],
                n_actions=logged.n_actions,
                logging_action_dist=None
                if logged.logging_action_dist is None
                else logged.logging_action_dist[idx],
                group_id=None if logged.group_id is None else logged.group_id[idx],
            ),
            position=self.position[idx],
            timestamp=timestamps,
            user_features=users,
            item_feature_0=self.item_feature_0,
            item_feature_labels=self.item_feature_labels,
            provenance=self.provenance,
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def uniform_evaluation_probabilities(data: OpenBanditData) -> FloatArray:
    """``pi(a | row) = 1/K`` for every row and action. Rows are not renormalised."""
    return np.full((data.n_samples, data.n_actions), 1.0 / data.n_actions)


def constant_action_probabilities(data: OpenBanditData, action: int) -> FloatArray:
    """Deterministic policy that selects one item at every row."""
    if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < data.n_actions:
        raise ConfigurationError(
            f"Constant action must be an int in [0, {data.n_actions - 1}], got {action!r}."
        )
    dist = np.zeros((data.n_samples, data.n_actions), dtype=np.float64)
    dist[:, action] = 1.0
    return dist


def load_open_bandit(
    obd_root: Path | str,
    *,
    behavior_policy: str,
    campaign: str,
) -> OpenBanditData:
    """Load one small-sample behavior file and its item table.

    ``obd_root`` is the ``obd/`` directory that contains ``random/`` and ``bts/``.
    """
    if behavior_policy not in BEHAVIOR_POLICIES:
        raise ConfigurationError(
            f"`behavior_policy` must be one of {list(BEHAVIOR_POLICIES)}, got {behavior_policy!r}."
        )
    if campaign not in CAMPAIGNS:
        raise ConfigurationError(f"`campaign` must be one of {list(CAMPAIGNS)}, got {campaign!r}.")
    root = Path(obd_root)
    directory = root / behavior_policy / campaign
    behavior_path = directory / f"{campaign}.csv"
    item_path = directory / "item_context.csv"
    if not behavior_path.is_file() or not item_path.is_file():
        raise ConfigurationError(
            f"Expected {behavior_path} and {item_path}. "
            "This loader reads the small sample shipped in st-tech/zr-obp under obd/. "
            "It does not download the full Open Bandit Dataset."
        )
    behavior_rows, behavior_fields = _read_csv(behavior_path)
    _require_columns(behavior_path, behavior_fields, _behavior_columns_before_affinity())
    if PROBABILITY_COLUMN not in behavior_fields and "action_prob" in behavior_fields:
        raise DataValidationError(
            f"{behavior_path} has `action_prob` and not `{PROBABILITY_COLUMN}`. "
            "The zr-obp README uses the name action_prob, but the small-sample files "
            f"and OpenBanditDataset.load_raw_data use `{PROBABILITY_COLUMN}`. "
            "Rename the column explicitly if you intend them to be the same quantity. "
            "This loader will not alias them."
        )
    item_rows, item_fields = _read_csv(item_path)
    _require_columns(
        item_path,
        item_fields,
        ("item_id", "item_feature_0", "item_feature_1", "item_feature_2", "item_feature_3"),
    )
    return _assemble(
        behavior_path,
        behavior_rows,
        behavior_fields,
        item_path,
        item_rows,
        behavior_policy,
        campaign,
    )


def _behavior_columns_before_affinity() -> tuple[str, ...]:
    return (
        "timestamp",
        "item_id",
        "position",
        "click",
        PROBABILITY_COLUMN,
        *USER_FEATURE_COLUMNS,
    )


def _read_csv(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DataValidationError(f"{path} has no header.")
        fields = tuple(name for name in reader.fieldnames if name)
        rows = list(reader)
    if not rows:
        raise DataValidationError(f"{path} has a header and no data rows.")
    return rows, fields


def _require_columns(path: Path, fields: tuple[str, ...], required: tuple[str, ...]) -> None:
    missing = [name for name in required if name not in fields]
    if missing:
        raise DataValidationError(
            f"{path} is missing column(s) {missing}. Found {list(fields)[:12]}"
            f"{'...' if len(fields) > 12 else ''}."
        )


def _assemble(
    behavior_path: Path,
    rows: list[dict[str, str]],
    fields: tuple[str, ...],
    item_path: Path,
    item_rows: list[dict[str, str]],
    behavior_policy: str,
    campaign: str,
) -> OpenBanditData:
    n = len(rows)
    action = _int_column(behavior_path, rows, "item_id")
    position = _int_column(behavior_path, rows, "position")
    click = _int_column(behavior_path, rows, "click")
    propensity = _float_column(behavior_path, rows, PROBABILITY_COLUMN)
    if np.any((position < 1) | (position > 3)):
        bad = int(np.flatnonzero((position < 1) | (position > 3))[0])
        raise DataValidationError(
            f"{behavior_path} row {bad} has position {int(position[bad])}. "
            "Verified small-sample positions are 1 (left), 2 (center), and 3 (right). "
            "The value was not recoded."
        )
    if not np.all((click == 0) | (click == 1)):
        bad = int(np.flatnonzero((click != 0) & (click != 1))[0])
        raise DataValidationError(
            f"{behavior_path} row {bad} has click {int(click[bad])}. Clicks must be 0 or 1."
        )
    if np.any(action < 0):
        raise DataValidationError(f"{behavior_path} has a negative item_id.")
    n_actions = int(np.max(action)) + 1
    present = set(int(a) for a in action)
    missing_ids = [i for i in range(n_actions) if i not in present]
    if missing_ids:
        shown = missing_ids[:8]
        raise DataValidationError(
            f"{behavior_path} item_id values are not the contiguous index 0..{n_actions - 1}. "
            f"Missing {shown}. Item ids are not remapped, because item_context.csv is "
            "indexed by the same integers."
        )
    affinity_names = tuple(f"user-item_affinity_{i}" for i in range(n_actions))
    _require_columns(behavior_path, fields, affinity_names)
    affinity = np.column_stack(
        [_float_column(behavior_path, rows, name) for name in affinity_names]
    )
    users: list[tuple[str, str, str, str]] = []
    for row in rows:
        user = tuple(row[column] for column in USER_FEATURE_COLUMNS)
        if len(user) != 4:
            raise DataValidationError(f"{behavior_path} is missing a user feature column.")
        users.append((user[0], user[1], user[2], user[3]))
    if any(part == "" for user in users for part in user):
        raise DataValidationError(f"{behavior_path} has an empty user_feature value.")
    timestamps = tuple(rows[i]["timestamp"] for i in range(n))
    if any(stamp == "" for stamp in timestamps):
        raise DataValidationError(f"{behavior_path} has an empty timestamp.")

    item_feature_0, item_labels = _item_table(item_path, item_rows, n_actions)
    uniform = _uniform_logging_distribution(propensity, n_actions)
    group_id = np.array(["|".join(user) for user in users])
    logged = LoggedBanditData(
        context=affinity,
        action=action,
        reward=click.astype(np.float64),
        logging_propensity=propensity,
        n_actions=n_actions,
        logging_action_dist=uniform,
        group_id=group_id,
    )
    counts: dict[str, int] = {}
    for stamp in timestamps:
        counts[stamp] = counts.get(stamp, 0) + 1
    repeated = sum(1 for count in counts.values() if count > 1)
    provenance = OpenBanditProvenance(
        behavior_policy=behavior_policy,
        campaign=campaign,
        behavior_path=str(behavior_path),
        behavior_sha256=_sha256_file(behavior_path),
        item_context_path=str(item_path),
        item_context_sha256=_sha256_file(item_path),
        n_rows=n,
        n_actions=n_actions,
        logging_distribution="uniform_identified"
        if uniform is not None
        else "observed_action_only",
        n_repeated_timestamps=repeated,
        max_rows_per_timestamp=max(counts.values()),
        n_user_feature_tuples=len(set(users)),
    )
    return OpenBanditData(
        logged=logged,
        position=position,
        timestamp=timestamps,
        user_features=tuple(users),
        item_feature_0=item_feature_0,
        item_feature_labels=item_labels,
        provenance=provenance,
    )


def _uniform_logging_distribution(propensity: FloatArray, n_actions: int) -> FloatArray | None:
    """Return the full uniform distribution when the file identifies it, else None."""
    target = 1.0 / n_actions
    if np.allclose(propensity, target, rtol=0.0, atol=1e-12):
        return np.full((propensity.shape[0], n_actions), target)
    return None


def _item_table(
    path: Path, rows: list[dict[str, str]], n_actions: int
) -> tuple[FloatArray, tuple[tuple[str, str, str], ...]]:
    ids = _int_column(path, rows, "item_id")
    if sorted(int(i) for i in ids) != list(range(n_actions)):
        raise DataValidationError(
            f"{path} item_id values are not 0..{n_actions - 1}, which is the action index "
            "of the behavior file. The tables were not aligned by sorting or filling gaps."
        )
    feature_0 = np.empty(n_actions, dtype=np.float64)
    labels: list[tuple[str, str, str]] = [("", "", "")] * n_actions
    values = _float_column(path, rows, "item_feature_0")
    for row_index, item_id in enumerate(ids):
        slot = int(item_id)
        feature_0[slot] = values[row_index]
        labels[slot] = (
            rows[row_index]["item_feature_1"],
            rows[row_index]["item_feature_2"],
            rows[row_index]["item_feature_3"],
        )
    if any(any(part == "" for part in triple) for triple in labels):
        raise DataValidationError(f"{path} has an empty item feature label.")
    return feature_0, tuple(labels)


def _int_column(path: Path, rows: list[dict[str, str]], name: str) -> IntArray:
    parsed: list[int] = []
    for index, row in enumerate(rows):
        text = row.get(name, "")
        try:
            value = int(text)
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                f"{path} row {index} column `{name}` is {text!r}. An integer is required. "
                "The value was not rounded or dropped."
            ) from exc
        if text != str(value):
            raise DataValidationError(
                f"{path} row {index} column `{name}` is {text!r}, not a canonical integer."
            )
        parsed.append(value)
    return np.asarray(parsed, dtype=np.int64)


def _float_column(path: Path, rows: list[dict[str, str]], name: str) -> FloatArray:
    parsed = np.empty(len(rows), dtype=np.float64)
    for index, row in enumerate(rows):
        text = row.get(name, "")
        try:
            value = float(text)
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                f"{path} row {index} column `{name}` is {text!r}. A finite number is required."
            ) from exc
        if not math.isfinite(value):
            raise DataValidationError(
                f"{path} row {index} column `{name}` is non-finite ({text!r})."
            )
        parsed[index] = value
    return parsed
