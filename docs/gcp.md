# Deploying on Google Cloud

This guide is for engineers putting the ForensiX server (API and web manager) on Google Cloud.
Pick a target, then deploy it from Cloud Shell, with one command, or with one click in GitHub
Actions. The matching teardown removes everything the deploy created.

| Target | What runs | Pick it when |
|---|---|---|
| **`cloudrun`**: Cloud Run, CPU | Serverless containers, HTTPS included, scale on requests | The simplest service, moderate volume, mostly images and audio |
| **`cloudrun-gpu`**: Cloud Run, GPU | One NVIDIA L4 per instance | Video and audiovisual models without managing anything |
| **`gke`**: GKE Autopilot | Kubernetes, nodes Google manages and scales | You run Kubernetes already, or want the same manifests as on other clouds |
| **`gke-gpu`**: GKE Autopilot, GPU | Kubernetes, an L4 node per pod, created on demand | Kubernetes plus GPUs |

| File | What it is |
|---|---|
| [`deploy/gcp/deploy.sh`](../deploy/gcp/deploy.sh) | One command to deploy or tear down any target |
| [`deploy/kubernetes/overlays/gke`](../deploy/kubernetes/overlays/gke/), [`gke-gpu`](../deploy/kubernetes/overlays/gke-gpu/) | The [Kubernetes manifests](cloud.md#kubernetes) adapted to GKE Autopilot |
| [`.github/workflows/deploy-gcp.yml`](../.github/workflows/deploy-gcp.yml) | The same, as a one-click GitHub Actions workflow |

## Before you start: the image

Every target runs `ghcr.io/anas-rz/forensix:cpu-bundled` or `:gpu-bundled`, which have the
default models baked in. Push a version tag (`git tag v0.1.0 && git push origin v0.1.0`) to
publish them, then make the GHCR package public ([details](aws.md#before-you-start-the-image)).

- **GKE** pulls from GHCR directly.
- **Cloud Run** only pulls from Artifact Registry (and Docker Hub), so the script creates an
  Artifact Registry *remote repository* named `ghcr` that proxies ghcr.io and caches the image
  in your region.

To use your own image instead, push it to Artifact Registry and pass `--image`.

## Option 1: from Cloud Shell

[![Open in Cloud Shell](https://gstatic.com/cloudssh/images/open-btn.svg)](https://ssh.cloud.google.com/cloudshell/open?cloudshell_git_repo=https://github.com/anas-rz/ForensiX)

The button opens Cloud Shell with this repository cloned. `gcloud` and `kubectl` are already
installed and signed in, so run:

```bash
gcloud config set project YOUR_PROJECT
deploy/gcp/deploy.sh cloudrun          # or cloudrun-gpu, gke, gke-gpu
```

## Option 2: one command

Needs the [gcloud CLI](https://cloud.google.com/sdk/docs/install), signed in with a project
selected. GKE also needs `kubectl` and `gke-gcloud-auth-plugin`, installed with
`gcloud components install kubectl gke-gcloud-auth-plugin`.

```bash
deploy/gcp/deploy.sh cloudrun                     # or cloudrun-gpu, gke, gke-gpu
deploy/gcp/deploy.sh cloudrun --destroy           # delete everything it created
```

| Option | Default | Meaning |
|---|---|---|
| `--name` | `forensix` | Prefix for services, the cluster, the bucket and the keys |
| `--project` | gcloud's current project | Project to deploy into |
| `--region` | `us-central1` | Region. GPU targets need an L4 region, e.g. `us-central1`, `us-east4`, `europe-west1`, `europe-west4` or `asia-southeast1` |
| `--image` | `ghcr.io/anas-rz/forensix:{cpu,gpu}-bundled` | Your image |

Rerunning the deploy updates in place. It prints the URLs and how to read the keys, which are
generated once in Secret Manager and never printed:

```bash
gcloud secrets versions access latest --secret forensix-api-key      # and forensix-admin-key
curl -H "X-API-Key: $KEY" -F file=@photo.jpg https://<inference URL>/v1/detect
```

## Option 3: one click in GitHub Actions

**Actions → deploy-gcp → Run workflow**: pick the target, `deploy` or `destroy`, the name and the
region. The URLs and key commands appear in the run summary.

**One-time setup:** GitHub signs in with Workload Identity Federation, so no service account keys
are stored.

```bash
PROJECT=your-project; NUMBER=$(gcloud projects describe $PROJECT --format 'value(projectNumber)')
gcloud iam workload-identity-pools create github --project $PROJECT --location global
gcloud iam workload-identity-pools providers create-oidc github --project $PROJECT --location global \
  --workload-identity-pool github --issuer-uri https://token.actions.githubusercontent.com \
  --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition "assertion.repository == 'anas-rz/ForensiX'"
gcloud iam service-accounts create forensix-deploy --project $PROJECT
gcloud projects add-iam-policy-binding $PROJECT --role roles/owner \
  --member serviceAccount:forensix-deploy@$PROJECT.iam.gserviceaccount.com
gcloud iam service-accounts add-iam-policy-binding forensix-deploy@$PROJECT.iam.gserviceaccount.com \
  --project $PROJECT --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/projects/$NUMBER/locations/global/workloadIdentityPools/github/attribute.repository/anas-rz/ForensiX"
```

A deploy enables APIs and creates service accounts, buckets, secrets, services or clusters, so the
account needs broad rights. In a shared project, replace `roles/owner` with Cloud Run Admin,
Kubernetes Engine Admin, Storage Admin, Secret Manager Admin, Artifact Registry Admin, Service
Account Admin, Service Account User, Project IAM Admin and Service Usage Admin.

In the repository, create the environment `gcp` (**Settings → Environments**; add required
reviewers to gate deploys). Then add these repository secrets:
- `GCP_WORKLOAD_IDENTITY_PROVIDER` = `projects/NUMBER/locations/global/workloadIdentityPools/github/providers/github`
- `GCP_SERVICE_ACCOUNT` = `forensix-deploy@PROJECT.iam.gserviceaccount.com`
- `GCP_PROJECT_ID`

## What gets deployed

**Cloud Run** (`cloudrun`, `cloudrun-gpu`):

- **`<name>-manager`** serves the web manager, dataset uploads, calibration and fairness jobs.
  It runs as exactly one instance with always-on CPU, so background jobs keep running between
  requests, and it's the single writer of shared state.
- **`<name>-inference`** serves `/v1/detect`. It scales from 1 to 10 instances (1 to 4 on GPU) on
  concurrent requests and can't write shared state.
- **Two URLs.** Each service has its own HTTPS URL on `run.app`. Open the web manager on the
  manager URL and send API traffic to the inference URL. (The manager also answers `/v1/detect`,
  which the web manager's Test page uses.) To put both behind one domain, add an external
  Application Load Balancer with serverless network endpoint groups and route the management
  paths (see the [Ingress](cloud.md#what-gets-deployed)) to the manager.
- **Shared state:** a Cloud Storage bucket, `<project>-<name>-data`, mounted at `/data` in both
  services. An activated calibration applies to the inference service on its next request.
- **Sizing:** 4 vCPUs and 16 GiB per instance on CPU; 8 vCPUs, 32 GiB and an L4 on GPU. A startup
  probe on `/ready` waits up to 15 minutes for the models to load.
- **Identity and keys:** both services run as a dedicated service account, `<name>-run`, that can
  read only the two keys and the bucket.

**GKE Autopilot** (`gke`, `gke-gpu`):

- **Cluster:** a regional Autopilot cluster. Google provisions nodes for the pods the HPA asks
  for; `gke-gpu` pods each get an L4 node.
- **Workloads:** the `gke` or `gke-gpu` overlay:
  - a Filestore volume (the `standard-rwx` class, 1 TiB minimum);
  - the GKE Ingress, which creates an external Application Load Balancer;
  - a `BackendConfig` with a 900-second timeout and health checks on `/ready`.
- **Keys:** generated into Secret Manager (`<name>-api-key`, `<name>-admin-key`) and copied into
  the `forensix-keys` Kubernetes Secret.

## HTTPS and a domain

- **Cloud Run:** HTTPS is on by default on the `run.app` URLs. For your own domain, use Cloud Run
  domain mappings or a load balancer.
- **GKE:** the Ingress serves plain HTTP on its IP. Point a host at that IP, then add a
  `ManagedCertificate` and the `networking.gke.io/managed-certificates` annotation to your copy of
  the overlay. Until then, restrict access or don't use real keys over it.

Cloud Run services allow unauthenticated invocation, so anyone can reach them, while ForensiX's
own keys protect the API. If an organization policy forbids public services, redeploy with
`--no-allow-unauthenticated` and call them with Google identity tokens.

## Costs and teardown

- **Cloud Run:** you pay per instance-second. The manager and one inference instance stay on, and
  GPU instances are billed while they run.
- **GKE:** Autopilot bills pod resources, plus the cluster fee, the load balancer and Filestore.
  The 1 TiB Filestore minimum is a fixed monthly cost.

`--destroy` (or the workflow's `destroy` action) deletes the services or cluster, the bucket or
Filestore volume with your datasets and calibrations, and the keys. The `ghcr` proxy repository is
kept, since other deployments may share it. Delete it with
`gcloud artifacts repositories delete ghcr --location REGION`.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Cloud Run: `Image ... not found` | The GHCR image isn't published or isn't public, so the proxy can't fetch it: see [the image](#before-you-start-the-image) |
| Cloud Run GPU: quota error | Request *Total Nvidia L4 GPU allocation without zonal redundancy* quota for Cloud Run in the region |
| GKE GPU pods stay `Pending` | No L4 capacity or quota in the region; `kubectl describe pod` shows the scale-up event |
| GKE Ingress has no IP, or returns 502 for a while | Load balancer provisioning takes about 5–10 minutes after the pods are ready |
| `POST /v1/datasets` returns 409 | You sent it to the inference service; use the manager URL |
| Logs | `gcloud run services logs tail <name>-manager --region REGION`, or `kubectl -n forensix logs deploy/forensix-manager` |

## What's been verified

- The deploy script passes ShellCheck.
- The `gke` and `gke-gpu` overlay patches apply cleanly to the base manifests.

Deployments to a Google Cloud project haven't been run from this repository yet. Try a
non-production project first.

**Next step:** open the web manager's [dashboard](ui.md#the-overview-dashboard), then [calibrate](calibration.md) on your media.
