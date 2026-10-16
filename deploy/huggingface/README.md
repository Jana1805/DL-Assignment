---
title: JIT Commit Risk API
emoji: 🛡️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
---

# JIT Commit Risk API

Scores how likely a Git commit is to introduce a bug, from 12 change metrics.

- `GET /health` returns the status and model version.
- `POST /predict` returns `risk_score`, `is_risky`, `threshold` and `model_version`.
- `GET /docs` shows the request schema.

The score is a review hint, not a merge gate: expect many false alarms.
