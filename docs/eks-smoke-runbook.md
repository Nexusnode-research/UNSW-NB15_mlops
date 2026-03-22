# EKS Smoke-Test Runbook — UNSW-NB15_MLOPS

**Goal:** cheapest valid EKS proof — cluster up, both pods running, `/health` OK,
UI loads, one authenticated prediction works, cluster torn down the same day.

**What this is NOT:** a permanent environment, a production deployment, or a
resurrection of the old full AWS setup. No NAT gateways, no LoadBalancer,
no ingress, no ALB/NLB, no Fargate.

---

## Prerequisites

| Tool | Min version | Check |
|---|---|---|
| `aws` CLI | any recent | `aws sts get-caller-identity` |
| `eksctl` | ≥ 0.200 | `eksctl version` |
| `kubectl` | ≥ 1.28 | `kubectl version --client` |
| `docker` | any | `docker version` |

```powershell
# Confirm identity and region
aws sts get-caller-identity
aws configure get region   # should be af-south-1
```

---

## Step 1 — Create ECR repositories (one-time)

```powershell
$REGION = "af-south-1"
$ACCOUNT = "002232587129"

aws ecr create-repository `
  --repository-name unsw-nb15-mlops-api `
  --region $REGION

aws ecr create-repository `
  --repository-name unsw-nb15-mlops-ui `
  --region $REGION

# Confirm
aws ecr describe-repositories `
  --query "repositories[].repositoryUri" `
  --output table
```

---

## Step 2 — Build and push images

```powershell
$REGION  = "af-south-1"
$ACCOUNT = "002232587129"
$REG     = "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
$TAG     = "smoke"

# IMPORTANT: hydrate LFS files before building.
# scaler.pkl and xgb.onnx are tracked by Git LFS.
# Without this step, Docker will COPY empty LFS pointer text files
# and the API will fail at startup with a bundle validation error.
git lfs pull

# Confirm the real files are present (not LFS pointers)
# scaler.pkl should be > 1 KB; xgb.onnx should be ~1.9 MB
Get-Item notebooks/ids_unsw/models/scaler.pkl | Select-Object Name, Length
Get-Item notebooks/ids_unsw/models/bundle_xgb/xgb.onnx | Select-Object Name, Length

# Authenticate Docker to ECR
aws ecr get-login-password --region $REGION |
  docker login --username AWS --password-stdin $REG

# Build (force linux/amd64 — EKS nodes are x86_64)
docker build --platform linux/amd64 `
  -f Dockerfile.api `
  -t "${REG}/unsw-nb15-mlops-api:${TAG}" `
  .

docker build --platform linux/amd64 `
  -f Dockerfile.ui `
  -t "${REG}/unsw-nb15-mlops-ui:${TAG}" `
  .

# Push
docker push "${REG}/unsw-nb15-mlops-api:${TAG}"
docker push "${REG}/unsw-nb15-mlops-ui:${TAG}"
```

> **Note:** if you are on an ARM Mac/Windows ARM, the `--platform linux/amd64`
> flag is required. On a Windows x86_64 machine, it is optional but harmless.

---

## Step 3 — Create the EKS cluster

```powershell
# ~15 minutes. Cluster + managed nodegroup in public subnets, no NAT gateway.
eksctl create cluster -f infra/eks-smoke-cluster.yaml

# On completion, eksctl updates your kubeconfig automatically.
# Verify:
kubectl get nodes
# Expected: 2 nodes in Ready state
```

If cluster creation fails part-way:

```powershell
# Check CloudFormation stacks
aws cloudformation describe-stacks `
  --query "Stacks[?contains(StackName,'unsw-mlops-smoke')].[StackName,StackStatus]" `
  --output table

# Check eksctl logs for the specific error, then:
eksctl delete cluster --name unsw-mlops-smoke --region af-south-1
# Fix the issue before retrying
```

---

## Step 4 — Set up namespace and Secret

```powershell
# Create the namespace
kubectl create namespace ids

# Create the secret (replace the value with a real token)
$TOKEN = "replace-with-a-long-random-string-at-least-32-chars"

kubectl create secret generic ids-api-secrets `
  --namespace ids `
  --from-literal=IDS_API_TOKEN=$TOKEN

# Verify
kubectl get secret ids-api-secrets -n ids
```

---

## Step 5 — Deploy

```powershell
kubectl apply -k k8s/overlays/eks-smoke

# Watch rollout
kubectl -n ids rollout status deploy/ids-api   --timeout=120s
kubectl -n ids rollout status deploy/dash-ui   --timeout=120s

# Confirm pods are Running
kubectl get pods -n ids -w
# Expected:
#   ids-api-xxxxx    1/1   Running
#   dash-ui-xxxxx    1/1   Running
```

If a pod is stuck in `Pending` or `ErrImagePull`:

```powershell
kubectl describe pod -n ids <pod-name>   # check Events section

# Common causes:
# - ErrImagePull / ImagePullBackOff → ECR auth issue or wrong image tag
#   Fix: confirm the ECR image exists, re-push if needed
# - Pending / Insufficient CPU → node group not yet ready
#   Fix: wait 2-3 minutes, check kubectl get nodes
# - CrashLoopBackOff → bundle validation failed at startup
#   Fix: kubectl logs -n ids <pod-name> --previous
```

---

## Step 6 — Smoke-test verification

All access is via `kubectl port-forward` — no public endpoints.

### 6a. API health check (no auth)

```powershell
kubectl port-forward -n ids svc/ids-api 8000:8000
# (leave this terminal open)
```

In a second terminal:

```powershell
$TOKEN = "replace-with-a-long-random-string-at-least-32-chars"

# /health — public, no token needed
curl http://localhost:8000/health
# Expected: {"status":"ok","threshold":0.757143,...}

# /features — requires auth
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/features
# Expected: {"features":["dur","proto",...]}  (34 items)
```

### 6b. One authenticated prediction

```powershell
# Build a zero-valued payload from the live feature list
$FEATURES_JSON = curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8000/features
$FEATURES = ($FEATURES_JSON | ConvertFrom-Json).features

$PAYLOAD = @{
  instances = @(@($FEATURES | ForEach-Object { @{$_ = 0.0} } | Measure-Object) |
    ForEach-Object { $FEATURES | ForEach-Object -Begin { $h = @{} } -Process { $h[$_] = 0.0 } -End { $h } })
} | ConvertTo-Json -Depth 5

$PAYLOAD | curl -s -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer $TOKEN" `
  -d @-
# Expected: {"probabilities":[...],"predictions":[0 or 1],"threshold":0.757143}
```

Or more simply with Python:

```powershell
python - << 'PY'
import json, urllib.request, os

TOKEN = os.environ.get("IDS_API_TOKEN", "replace-with-a-long-random-string-at-least-32-chars")
BASE  = "http://localhost:8000"
AUTH  = {"Authorization": f"Bearer {TOKEN}"}

def get(path):
    req = urllib.request.Request(BASE + path, headers=AUTH)
    with urllib.request.urlopen(req) as r:
        return json.load(r)

# Health (public)
print("HEALTH:", get("/health")["status"])

# Features
feats = get("/features")["features"]
print(f"FEATURES: {len(feats)} features")

# Predict
payload = json.dumps({"instances": [{f: 0.0 for f in feats}]}).encode()
req = urllib.request.Request(
    BASE + "/predict", data=payload,
    headers={**AUTH, "Content-Type": "application/json"},
    method="POST"
)
with urllib.request.urlopen(req) as r:
    result = json.load(r)
print("PREDICTION:", result["predictions"], " THRESHOLD:", result["threshold"])
PY
```

### 6c. Dash UI check

```powershell
# Stop port-forward on 8000 first, or use a different local port
kubectl port-forward -n ids svc/dash-ui 8050:8050
```

Open `http://localhost:8050` in a browser.
Expected: Dash UI loads, shows the feature form and health status.

---

## Step 7 — Acceptance criteria

Check each before tearing down:

- [ ] `kubectl get nodes` shows 2 nodes in `Ready` state
- [ ] Both pods are `Running` (`kubectl get pods -n ids`)
- [ ] `GET /health` returns `{"status":"ok"}` without auth header
- [ ] `GET /features` returns 34 features with valid Bearer token
- [ ] `POST /predict` returns `predictions` and `probabilities` with valid payload
- [ ] `http://localhost:8050` returns HTTP 200 (Dash UI loads)

---

## Step 8 — Teardown (same day)

```powershell
# Delete all Kubernetes resources first (optional — cluster delete handles this)
kubectl delete namespace ids

# Delete the EKS cluster and all associated AWS resources
# (VPC, subnets, security groups, node group, IAM roles)
eksctl delete cluster --name unsw-mlops-smoke --region af-south-1

# This takes ~10 minutes. Monitor:
aws cloudformation describe-stacks `
  --query "Stacks[?contains(StackName,'unsw-mlops-smoke')].[StackName,StackStatus]" `
  --output table
# Wait until all stacks show DELETE_COMPLETE or disappear

# Confirm no running nodes
aws ec2 describe-instances `
  --filters "Name=tag:alpha.eksctl.io/cluster-name,Values=unsw-mlops-smoke" `
  --query "Reservations[].Instances[].InstanceId" `
  --output text
# Expected: (empty)
```

> **Keep the ECR repositories.** They cost ~$0.10/GB/month and the images are needed
> for future smoke tests. To delete them too:
> ```powershell
> aws ecr delete-repository --repository-name unsw-nb15-mlops-api --force --region af-south-1
> aws ecr delete-repository --repository-name unsw-nb15-mlops-ui --force --region af-south-1
> ```

---

## Cost estimate

| Resource | Duration | Cost (af-south-1) |
|---|---|---|
| 2x t3.medium nodes | ~4 hours | ~$0.20–0.30 |
| EKS control plane | ~4 hours | ~$0.40 |
| ECR storage (2 images ~500MB) | ongoing | ~$0.05/month |
| Data transfer (image pulls) | one-time | ~$0.01 |
| **Total same-day** | | **< $1.00** |

No NAT gateways ($0), no load balancers ($0), no persistent volumes ($0).

---

## Diff from eks-dev overlay

| Aspect | `eks-dev` | `eks-smoke` |
|---|---|---|
| dash-ui Service | `LoadBalancer` | `ClusterIP` |
| ids-api Service | `ClusterIP` | `ClusterIP` |
| HPA | Yes (1–5 replicas) | No |
| Replicas | 2 (dash-ui) | 1 each |
| Image tag | `latest` | `smoke` |
| Access | AWS LoadBalancer DNS | `kubectl port-forward` |
| Resource limits | Production-sized | Half-sized |
