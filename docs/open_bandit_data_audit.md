# Open Bandit Dataset audit

Status: small sample only, read from a local checkout of `st-tech/zr-obp` at commit `8cbd5fa4558b7ad2ba4781546d6604e4cc3e07c4`. The full log was not downloaded.

## Sources

| Source | What was read | What it is allowed to decide |
|---|---|---|
| `data/raw/zr-obp/obd/{random,bts}/{all,men,women}/{campaign}.csv` | The six behavior files, 10,000 data rows each | Column names, types, action ids, positions, propensities, clicks |
| `data/raw/zr-obp/obd/.../item_context.csv` | The six item tables beside those files | Item-feature values actually stored |
| `data/raw/zr-obp/obd/README.md` in the same commit | Field descriptions and the citation request | A claim only where it agrees with the files |
| `data/raw/zr-obp/obp/dataset/real.py` | `OpenBanditDataset.load_raw_data` and `pre_process` | What OBP does with the same files. Our adapter does not call it |
| `data/raw/zr-obp/LICENSE` and `pyproject.toml` | Apache-2.0, package version 0.5.5 | License of the library and of the small `obd/` files shipped in that repository |
| https://research.zozo.com/data.html | Not retrieved in this audit | License of the full ~26M-row release remains unchecked |

The checkout is gitignored (`data/raw/`). File identity is the SHA-256 below. ReversalLab's adapter is `src/reversallab/data/open_bandit.py`. It uses the csv module. It does not depend on pandas, PyTorch, or OBP.

## Behavior-file columns

Each behavior file has an unnamed first column (an empty header cell). `csv.DictReader` exposes it as `""`. OBP drops it with `index_col=0`. Our adapter ignores empty header names and does not use that column as an identifier.

Remaining columns, verified on all six files:

| Column | Stored type | Meaning taken from the file, then the README |
|---|---|---|
| `timestamp` | string | Impression time as text. Not parsed |
| `item_id` | canonical integer | Action index. See below |
| `position` | integer in `{1, 2, 3}` | Left, center, right on the ZOZOTOWN interface. README line 35. Not recoded |
| `click` | integer in `{0, 1}` | Stored as a float reward |
| `propensity_score` | float in `(0, 1]` | Probability of the displayed item at that position |
| `user_feature_0` … `user_feature_3` | non-empty strings | Hashed user features. Kept as strings |
| `user-item_affinity_0` … `user-item_affinity_{K-1}` | finite floats | One column per action. This is the matrix stored as `LoggedBanditData.context` |

The README (line 37) names the probability column `action_prob`. None of the six files has that column. They have `propensity_score`, which is also the name `OpenBanditDataset.load_raw_data` reads. The adapter rejects a file that has `action_prob` and not `propensity_score`. It does not alias the two names.

## Actions

`item_id` in every behavior file is the contiguous index `0 .. K-1`, with every integer present.

| Campaign | `K` | `item_id` |
|---|---|---|
| all | 80 | 0–79 |
| men | 34 | 0–33 |
| women | 46 | 0–45 |

README line 34 states those behavior-file ranges. README line 51 states different item-context ranges (`0-80`, `0-33`, `0-46`). The item tables in this checkout have the same ids as the behavior files: 80, 34, and 46 rows. There is no extra item. The adapter does not remap ids. A gap would be an error, because remapping would desynchronize `item_context.csv`.

## Positions

Raw positions are 1, 2, and 3. Counts in this sample:

| File | Position 1 | Position 2 | Position 3 |
|---|---:|---:|---:|
| random/all | 3322 | 3412 | 3266 |
| random/men | 3284 | 3388 | 3328 |
| random/women | 3329 | 3374 | 3297 |
| bts/all | 3362 | 3317 | 3321 |
| bts/men | 3339 | 3262 | 3399 |
| bts/women | 3288 | 3360 | 3352 |

OBP converts position with `(rankdata(position, "dense") - 1)` into `{0, 1, 2}` and can use that index as the third axis of an `(n, K, len_list)` action tensor. Our adapter stores 1, 2, and 3. Parity comparisons do not pass those values as OBP's position index. Each of our rows is already one position, so OBP receives `len_list = 1` and position index 0. See `docs/open_bandit_estimand.md`.

## Propensities

Random-policy files: one distinct value, equal to `1/K` (`0.0125`, `1/34`, `1/46`). Every action appears. The adapter's label for that case is `uniform_identified`, and it stores a uniform logging distribution. The file still does not contain a column for the actions that were not displayed. The uniform matrix is the random policy this constant column is consistent with. It is how support can be checked for the random logger.

BTS files: propensities vary by row. All are finite and in `(0, 1]`. The logging distribution on actions that were not displayed is not in the file. The adapter's label is `observed_action_only`, and `logging_action_dist` is `None`. Support cannot be verified. Diagnostics report `support_unverifiable`. IPS still runs. A zero propensity is rejected. A positive propensity that overflows the importance-weight ratio raises `NumericalError`. Neither case is repaired.

| File | Distinct propensities | Min | Max |
|---|---:|---:|---:|
| bts/all | 7883 | 4.5e-5 | 0.95424 |
| bts/men | 7814 | 1.65e-4 | 0.72529 |
| bts/women | 7874 | 1e-6 | 0.9628 |

## What a row is

README line 25 calls a row a user impression. The columns say otherwise: one `item_id`, one `position`, one `click`. In this sample a timestamp is not a three-slot impression.

| File | Timestamps that occur more than once | Maximum rows sharing a timestamp |
|---|---:|---:|
| random/all | 9 | 2 |
| random/men | 11 | 2 |
| random/women | 3 | 2 |
| bts/all | 1 | 2 |
| bts/men | 2 | 2 |
| bts/women | 1 | 2 |

There is no verified impression id and no slate id. The adapter does not invent one from the timestamp. User-feature 4-tuples do repeat (205–253 distinct tuples per 10,000 rows). `group_id` stores those four hashes joined by `|`. Estimators do not resample by it. Rows that share a user tuple are dependent, and an i.i.d. Wald interval ignores that.

Click rates in this sample are 0.0038 (random/all), 0.0046 (random/men, random/women, bts/women), 0.0042 (bts/all), and 0.0069 (bts/men). That is on the order of 40–70 clicks per file. These files are a schema and parity check. They are not a confidence-interval calibration.

## Context and item tables

OBP's default context is a one-hot encoding of columns whose names contain `user_feature`, with `drop_first=True`. It does not use the affinity block as that context. Our `logged.context` is the affinity matrix, shape `(N, K)`, columns `user-item_affinity_0` through `user-item_affinity_{K-1}` in that order. IPS and SNIPS do not read it when the candidate distribution is supplied as an array. The two context definitions are different. Parity does not depend on either one.

`item_feature_0` is a finite float and can be negative. For a fixed campaign and `item_id`, the random-policy and BTS copies of `item_feature_0` are the same string. `item_feature_1`, `item_feature_2`, and `item_feature_3` differ for every item between the random and BTS copies of the same campaign. They are stored as raw strings, indexed by `item_id`. They are not passed through OBP's `LabelEncoder`, and they are not joined across behavior policies.

## File order

OBP sorts by timestamp after `read_csv`. Our adapter keeps file order. Parity passes the adapter's arrays to both estimators. It does not assume that row `i` of our load is row `i` of `OpenBanditDataset`.

## Hashes of the files that were read

Behavior files:

| File | SHA-256 |
|---|---|
| random/all | `c0c58e3fa085803dd8dabfcfea77cf8d5fc0324e6ba6307528838d59324b7605` |
| random/men | `379fd2efc808b49c6082b81af6e322b69b2a267963e17a18dd27a12ed6e01571` |
| random/women | `12d8a65cb0a90823ea6b3fdb428501f2bd01534dbcea9e0876052d7b0d2ba2c5` |
| bts/all | `f27f426907a567ef658616263949536ff14e99a4c623ea9f343108b6699a06ac` |
| bts/men | `6459dfce78b37cb845b73099f07d2aa4ddb645a47e2332ca6998c0172e6879b8` |
| bts/women | `64697a19cc6519859019204dd3098a987b89f2b50d347f6a83311f8ecc7c036c` |

Item tables:

| File | SHA-256 |
|---|---|
| random/all | `912944b28f58d0823a7d1d4a914465c4597f59af4ada41e5d66e55e8cb0bb82b` |
| random/men | `42c4677b538168984bad569ea4a82b786923f22666a3013cad5a17ee183269ae` |
| random/women | `55a38c3bea6155020ca6b2d19d2897784bd0af69c604063b0a54d13702e21d5a` |
| bts/all | `275c6917a8de467b8a694b220abc2a1df1dfae8e0a6272aa1b3cb1c47930178a` |
| bts/men | `6d281ffaf16422b702577f415c77696078a666ced883a78cfa78626c97676b27` |
| bts/women | `4a90517d143682bb48ae73cc45e06a565d88d6416d3e71216620fd145884e79d` |

## License and attribution

The `zr-obp` repository, including the small `obd/` files in that repository, is Apache-2.0 (`LICENSE`, and `license` in `pyproject.toml`). Using the dataset requires the citation in `obd/README.md`: Saito, Aihara, Matsutani, and Narita, arXiv:2008.07146. The small files are not committed to ReversalLab.

The full dataset is distributed from https://research.zozo.com/data.html. That page was not fetched for this audit, so this document does not state the full-data license. Do not download or commit the full log until that license has been read.

## Uncertainties left open

* Whether two rows that share a timestamp are a partial slate, a collision, or a sample artifact. Multiplicity is at most 2 in these files, so they are not three-slot impressions.
* Whether repeated user-feature hashes are the same person. They are the only dependence key the files provide.
* Why hashed item features differ between behavior policies while `item_feature_0` does not. IPS does not use them.
* The full-data license, schema, and any published benchmark computed on the 26M-row log. A 10,000-row estimate is not that benchmark.
