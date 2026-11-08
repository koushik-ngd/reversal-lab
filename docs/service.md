# ReversalLab Evaluation Service (M6A)

The ReversalLab Evaluation Service exposes the core off-policy evaluation, paired policy comparison, and decision-reversal robustness engine via a minimal, strictly typed REST API built with FastAPI.

The service wraps the public SDK (`BanditDataset`, `PolicyValueEvaluator`, `PolicyComparison`) without altering or redefining any underlying mathematical estimator formulas or decision algorithms.

---

## 1. Installation & Optional Dependencies

The service dependencies (`fastapi`, `uvicorn`) are packaged as an optional extra (`service`) so that users of the core Python research SDK do not carry HTTP dependencies.

To install with service dependencies:

```bash
# With uv (recommended)
uv sync --extra service

# With pip
pip install "reversallab[service]"
```

---

## 2. Running the Local Service

Start the service locally on CPU:

```bash
uv run python -m uvicorn reversallab.service.app:app --host 127.0.0.1 --port 8000
```

Once running:
- Interactive OpenAPI docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Alternative ReDoc docs: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- OpenAPI schema JSON: [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)

---

## 3. Endpoints Reference

### 3.1 Health & Version
- `GET /health`: System liveness and health probe.
  ```json
  {
    "status": "healthy",
    "service": "reversallab",
    "timestamp": "2026-11-06T13:00:00Z"
  }
  ```
- `GET /version`: Semantic version identifiers for package, schema, and API.
  ```json
  {
    "version": "0.1.0rc1",
    "schema_version": "1.0.0",
    "api_version": "v1"
  }
  ```
- `GET /metrics`: In-memory operational metrics snapshot (request counts by endpoint/status, latencies, gate failure counters, process RSS memory).

---

### 3.2 Single-Policy Evaluation (`POST /v1/evaluate/policy`)

Evaluates the expected value $V(\pi) = \mathbb{E}[r(\pi)]$ using Inverse Propensity Scoring (IPS), Self-Normalized IPS (SNIPS), Direct Method (DM), or Doubly Robust (DR) estimation with leakage-safe cross-fitting.

#### Request Body
```json
{
  "dataset": {
    "contexts": [[0.1, 0.5], [-0.2, 0.8]],
    "actions": [0, 1],
    "rewards": [1.0, 0.0],
    "propensities": [0.5, 0.5],
    "n_actions": 2
  },
  "policy": {
    "type": "linear_softmax",
    "weights": [[0.5, -0.5], [0.2, -0.2]],
    "inverse_temperature": 1.0
  },
  "estimator": "dr",
  "cross_fitting_folds": 5,
  "confidence_level": 0.95,
  "seed": 42
}
```

#### Response Body
```json
{
  "request_id": "8f8b8098-b809-4bf4-b152-cbcefa006c09",
  "status": "success",
  "timestamp": "2026-11-06T13:00:01Z",
  "report": {
    "schema_version": "1.0.0",
    "policy_name": "evaluated_policy",
    "estimator": "dr",
    "point_estimate": 0.6241,
    "standard_error": 0.0312,
    "ci_lower": 0.5629,
    "ci_upper": 0.6853,
    "confidence_level": 0.95,
    "diagnostics": {
      "effective_sample_size": 284.1,
      "ess_fraction": 0.947,
      "max_weight_share": 0.007,
      "min_propensity": 0.5
    }
  }
}
```

---

### 3.3 Paired Policy Comparison & Decision Reversal (`POST /v1/evaluate/comparison`)

Runs paired policy difference estimation $\Delta = V(\pi_e) - V(\pi_b)$, evaluates operational diagnostic gates, searches for the minimum evidence-reversing demographic shift $d^*$, and computes forward worst-case robust mixture bounds.

#### Request Body
```json
{
  "dataset": {
    "contexts": [[0.1, 0.5], [-0.2, 0.8]],
    "actions": [0, 1],
    "rewards": [1.0, 0.0],
    "propensities": [0.5, 0.5],
    "n_actions": 2
  },
  "candidate_policy": {
    "type": "linear_softmax",
    "weights": [[0.5, -0.5], [0.2, -0.2]]
  },
  "baseline_policy": {
    "type": "uniform"
  },
  "margin": 0.02,
  "deploy_epsilon": 0.10,
  "max_epsilon": 0.40,
  "estimator": "dr",
  "cross_fitting_folds": 5,
  "confidence_level": 0.95,
  "min_propensity_threshold": 1e-4,
  "seed": 42
}
```

#### Response Body (Decision: RECOMMEND)
Returns HTTP 200 with `status: "success"` and the complete SDK report dictionary conforming to schema `1.0.0`.

#### Response Body on Diagnostic Gate Failure (Decision: INCONCLUSIVE)
When an operational reliability check fails (such as an active logging propensity deficit $0 < p_i < \tau$ on an actively selected action, or ESS collapse):
- **HTTP Status:** `200 OK` (the request is semantically valid; the empirical evidence is inconclusive).
- `status`: `"inconclusive"`
- `report.decision.outcome`: `"INCONCLUSIVE"`
- `report.diagnostics.gate_passed`: `false`
- `report.diagnostics.rejection_reasons`: Detailed diagnostic reason strings.

---

## 4. Input Validation & Error Semantics

Client errors and invalid requests return RFC 7807 problem details:

- **HTTP 422 (Unprocessable Content):**
  - Mismatched array lengths ($N$).
  - Action index out of bounds $a_i \notin [0, K)$.
  - Non-positive logging propensities ($p_i \le 0$).
  - Policy weights dimension mismatch ($(D, K) \neq (D_{\text{context}}, K_{\text{actions}})$).
  - Non-finite float values (`NaN`, `Inf`).
  - Pre-execution compute budget exceeded: $N \times \text{folds} \times K > 1{,}000{,}000$.
- **HTTP 413 (Payload Too Large):**
  - Enforced by ASGI streaming middleware even if `Content-Length` is absent.
  - Aborts requests larger than 10 MB (`10,485,760` bytes).
- **HTTP 500 (Internal Server Error):**
  - Sanitized response without exposing local file paths or Python tracebacks.

---

## 5. Security & Observability

- **Privacy & Redaction:** Context feature vectors, observed rewards, actions, and propensities are strictly excluded from logs.
- **Traceability:** Every request receives an `X-Request-ID` header (propagated or generated as UUID4), recorded in structured JSON access logs.
- **Sandboxing:** No local file path ingestion is supported. Datasets are transmitted directly in-memory as serialized arrays.
