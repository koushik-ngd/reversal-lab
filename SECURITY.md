# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1.0 | :x:                |

## Deployment Model & Security Boundary (Model A)

ReversalLab is designed and packaged under **Deployment Model A: Trusted-Local Single-User / Single-Tenant Execution**.

### Security Assumptions & Operating Environment
- **Loopback & Private Network Default**: The ReversalLab HTTP API service binds by default to `127.0.0.1` or runs inside an isolated container network.
- **No Built-in Authentication**: The service does not implement multi-tenant user accounts, OAuth2, or API key authentication. It is intended for researcher workstations, local CI pipelines, or private batch processing environments.
- **Operator Responsibilities for Public Exposure**: If this service is exposed to a multi-user, untrusted, or public network, the hosting operator **MUST** terminate TLS, provide authentication (e.g., mTLS, reverse-proxy API gateway, or OAuth2 proxy), restrict network access, and enforce rate limiting.
- **Local File System Access**: Persistent background jobs store inputs, results, and SQLite metadata in local state directories (default `./var/reversallab`). File permissions are restricted (`0o600` for files, `0o700` for directories on POSIX systems).

### Built-in Hardening Controls
- **Request Size Enforcement**: Synchronous and asynchronous endpoints enforce an incremental 10 MB streaming request size ceiling (`MAX_REQUEST_BYTES = 10 * 1024 * 1024`).
- **Resource Limits**: Concurrency is strictly bounded by `max_active_workers` (default 2 child worker processes) and an admission queue ceiling (`MAX_QUEUED_JOBS = 50`).
- **Worker Process Isolation**: Asynchronous evaluations run in isolated Python subprocesses with separate OS PIDs, preventing CPU starvation or memory leaks from affecting the FastAPI HTTP event loop.
- **Fencing & Advisory Locking**: Single-supervisor advisory file locking (`supervisor.lock`) prevents split-brain dispatch across multiple processes. Generation execution tokens fence against stale child worker commits after supervisor restarts or cancellations.
- **Strict Path Validation**: All job identifiers are validated against UUID4 specifications with path traversal verification (`resolve()` within storage boundaries).

## Reporting a Vulnerability

We take the security of ReversalLab seriously. If you believe you have discovered a vulnerability, please report it responsibly:

1. **Do not create a public GitHub issue.**
2. Submit a private vulnerability report via [GitHub Security Advisories](https://github.com/reversallab/reversallab/security/advisories) or email security concerns to `security@reversallab.org` (or contact project maintainers privately).
3. Include detailed steps to reproduce the issue, proof of concept code, and the affected version(s).
4. Project maintainers will acknowledge receipt within 48 hours and provide an estimated timeline for remediation.
