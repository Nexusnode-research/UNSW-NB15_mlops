# EKS Smoke Test Results (Pass 4)

**Date executed:** 2026-03-22
**Region:** `af-south-1`
**Cluster:** `unsw-mlops-smoke`
**Result:** **SUCCESS**

This document serves as the canonical proof of the end-to-end cloud deployment and verification of the UNSW-NB15_MLOPS system.

## 1. Local / CI Confidence
Prior to cloud deployment, the following local gates were verified (same checks as in the GitHub Actions CI workflow when run manually or on a PR to `main`; pushes alone do not trigger CI):
* **Bundle Validation:** `ids_unsw.validate_bundle` passed.
* **Unit Tests:** Canonical schema validation and legacy key rejection verified via `pytest`.
* **API Integration:** FastAPI test client verified all endpoint contracts (auth, payloads, responses).
* **Manifest Validation:** `kubectl kustomize k8s/overlays/eks-smoke` rendered successfully in CI/local runs.

## 2. Cloud Smoke Proof
The complete stack was deployed to AWS Elastic Kubernetes Service (EKS) and verified.

* **Cluster Creation:** Successfully provisioned `unsw-mlops-smoke` via `eksctl`.
* **Image Build & Push:** API and UI images built for `linux/amd64` and pushed to Amazon ECR with the `smoke` tag.
* **Manifest Rollout:** Custom `eks-smoke` overlay applied to the live cluster. Pods scaled up and reached `Running` state successfully.
* **API Health Check (`/health`):** Reached without authentication via `kubectl port-forward`, returning `{"status": "ok"}` and confirming the deployed probability threshold.
* **Feature Schema (`/features`):** Authenticated request successfully returned the expected 34 numerical features.
* **Prediction Contract (`/predict`):** Authenticated payload submitted and returned the correct prediction geometry (e.g., probability float and class `[0]`).
* **UI Load:** The Dash interface successfully returned HTTP 200 via a local port-forward proxy over gunicorn.

## 3. Teardown Proof
Following the successful validation, the environment was completely dismantled to prevent orphan costs:
* `eksctl delete cluster` executed cleanly.
* CloudFormation stacks containing the `smoke` string dropped to zero.
* EC2 instances bound to the cluster were successfully terminated.

## Minor Cleanup Notes for Future Improvement
While the proof is entirely valid, there are minor idempotency opportunities for future runbook refinement:
* **Namespace/Secret Creation:** Applying the `ids` namespace and `ids-api-secrets` via imperative commands (`create`) yielded "already exists" errors on reruns. These steps should transition to `kubectl apply`-style declarative idempotency.
* **Shell Workarounds:** The `aws` / `.py` Windows shell association problem required a `.bat` wrapper. This is a local host issue and doesn't affect the remote Linux deployment or CI.
* **Port-Forward Interruptions:** "Lost connection to pod" messages during port-forwarding were simply a consequence of the standard teardown/interruption flow locally, not an indicator of cluster health issues.

*Pass 4 is complete and the deployment pipeline is fully validated.*
