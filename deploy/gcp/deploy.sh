#!/usr/bin/env bash
# Deploy ForensiX to Google Cloud, or tear it down, with one command. See docs/gcp.md.
#
#   deploy/gcp/deploy.sh cloudrun            # Cloud Run, CPU: a manager and an autoscaled inference service
#   deploy/gcp/deploy.sh cloudrun-gpu        # Cloud Run with an NVIDIA L4 per instance
#   deploy/gcp/deploy.sh gke                 # GKE Autopilot cluster, CPU
#   deploy/gcp/deploy.sh gke-gpu             # GKE Autopilot cluster, NVIDIA L4 GPUs
#   deploy/gcp/deploy.sh cloudrun --destroy  # delete everything the deploy created
#
# Options: --name NAME (default forensix), --project PROJECT (default: gcloud's current project),
#          --region REGION (default us-central1; GPU targets need an L4 region),
#          --image IMAGE (default ghcr.io/anas-rz/forensix:{cpu,gpu}-bundled, through an Artifact
#          Registry proxy for Cloud Run).
# Needs: gcloud (authenticated); for GKE also kubectl and gke-gcloud-auth-plugin.
set -euo pipefail

usage() { sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

TARGET="${1:-}"; [[ "$TARGET" =~ ^(cloudrun|cloudrun-gpu|gke|gke-gpu)$ ]] || usage 1
shift
NAME=forensix PROJECT="" REGION="${GCP_REGION:-us-central1}" IMAGE="" DESTROY=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --destroy) DESTROY=true; shift ;;
    -h|--help) usage ;;
    *) echo "unknown option: $1" >&2; usage 1 ;;
  esac
done
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
log() { printf '\n==> %s\n' "$*"; }
need() { command -v "$1" >/dev/null || { echo "$1 is required (see docs/gcp.md)" >&2; exit 1; }; }
summary() { [[ -n "${GITHUB_STEP_SUMMARY:-}" ]] && printf '%s\n' "$@" >> "$GITHUB_STEP_SUMMARY"; printf '%s\n' "$@"; }

need gcloud
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
[[ -n "$PROJECT" ]] || { echo "No project: pass --project or run gcloud config set project PROJECT" >&2; exit 1; }
export CLOUDSDK_CORE_PROJECT="$PROJECT"
GPU=false; [[ "$TARGET" == *-gpu ]] && GPU=true
FLAVOR=cpu-bundled; $GPU && FLAVOR=gpu-bundled
API_SECRET="$NAME-api-key" ADMIN_SECRET="$NAME-admin-key"

# Keys live in Secret Manager: generated once, reused on every redeploy, never printed.
ensure_keys() {
  for s in "$API_SECRET" "$ADMIN_SECRET"; do
    if ! gcloud secrets describe "$s" >/dev/null 2>&1; then
      python3 -c 'import secrets; print(secrets.token_urlsafe(30), end="")' |
        gcloud secrets create "$s" --replication-policy=automatic --data-file=- >/dev/null
    fi
  done
}
delete_keys() { for s in "$API_SECRET" "$ADMIN_SECRET"; do gcloud secrets delete "$s" --quiet 2>/dev/null || true; done; }
key_help() { echo "\`gcloud secrets versions access latest --project $PROJECT --secret $1\`"; }

# -- Cloud Run: two services sharing a Cloud Storage bucket --------------------------------------
cloudrun() {
  local bucket="$PROJECT-$NAME-data" sa="$NAME-run@$PROJECT.iam.gserviceaccount.com"
  if $DESTROY; then
    log "Deleting the Cloud Run services, bucket gs://$bucket (datasets and calibrations), keys and service account"
    for svc in "$NAME-manager" "$NAME-inference"; do gcloud run services delete "$svc" --region "$REGION" --quiet 2>/dev/null || true; done
    gcloud storage rm -r "gs://$bucket" --quiet 2>/dev/null || true
    delete_keys
    gcloud iam service-accounts delete "$sa" --quiet 2>/dev/null || true
    summary "Deleted ForensiX on Cloud Run \`$NAME\` in $REGION ($PROJECT)."
    return
  fi
  log "Enabling APIs"
  gcloud services enable run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com storage.googleapis.com iam.googleapis.com
  if [[ -z "$IMAGE" ]]; then
    # Cloud Run pulls only from Artifact Registry (and Docker Hub): proxy GHCR through a remote repository
    if ! gcloud artifacts repositories describe ghcr --location "$REGION" >/dev/null 2>&1; then
      log "Creating Artifact Registry remote repository 'ghcr' (a pull-through proxy of ghcr.io)"
      gcloud artifacts repositories create ghcr --location "$REGION" --repository-format docker \
        --mode remote-repository --remote-repo-config-desc "GitHub Container Registry" --remote-docker-repo https://ghcr.io
    fi
    IMAGE="$REGION-docker.pkg.dev/$PROJECT/ghcr/anas-rz/forensix:$FLAVOR"
  fi
  ensure_keys
  if ! gcloud iam service-accounts describe "$sa" >/dev/null 2>&1; then
    gcloud iam service-accounts create "$NAME-run" --display-name "ForensiX on Cloud Run ($NAME)"
  fi
  if ! gcloud storage buckets describe "gs://$bucket" >/dev/null 2>&1; then
    gcloud storage buckets create "gs://$bucket" --location "$REGION" --uniform-bucket-level-access
  fi
  gcloud storage buckets add-iam-policy-binding "gs://$bucket" --member "serviceAccount:$sa" --role roles/storage.objectUser >/dev/null
  for s in "$API_SECRET" "$ADMIN_SECRET"; do
    gcloud secrets add-iam-policy-binding "$s" --member "serviceAccount:$sa" --role roles/secretmanager.secretAccessor >/dev/null
  done

  local size=(--cpu 4 --memory 16Gi) device=cpu
  if $GPU; then size=(--cpu 8 --memory 32Gi --gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy); device=cuda; fi
  local common=(--region "$REGION" --image "$IMAGE" --service-account "$sa" --port 8000 "${size[@]}"
    --execution-environment gen2 --cpu-boost --timeout 3600 --allow-unauthenticated
    --set-secrets "FORENSIX_SERVER_API_KEY=$API_SECRET:latest,FORENSIX_SERVER_ADMIN_KEY=$ADMIN_SECRET:latest"
    --add-volume "name=data,type=cloud-storage,bucket=$bucket,mount-options=uid=10001;gid=10001"
    --add-volume-mount "volume=data,mount-path=/data"
    --startup-probe "httpGet.path=/ready,httpGet.port=8000,periodSeconds=10,failureThreshold=90,timeoutSeconds=5")
  local envs; envs=$(mktemp)
  write_env() {  # $1 = admin endpoints and UI on (manager) or off (inference)
    cat > "$envs" <<EOF
FORENSIX_SERVER_DEVICE: "$device"
FORENSIX_SERVER_PRELOAD: "dfdet-clip commfor nes2net-x-itw syncnet c2pa watermark-image watermark-audio metadata"
FORENSIX_SERVER_CALIBRATION_DIR: "/data/calibration"
FORENSIX_SERVER_GALLERY_DIR: "/data/galleries"
FORENSIX_SERVER_ADMIN_ENDPOINTS: "$1"
FORENSIX_SERVER_UI: "$1"
LOG_FORMAT: "json"
EOF
  }
  log "Deploying $NAME-manager (web manager, datasets, calibration): one always-on instance"
  write_env true
  gcloud run deploy "$NAME-manager" "${common[@]}" --env-vars-file "$envs" \
    --min-instances 1 --max-instances 1 --no-cpu-throttling --concurrency 8
  log "Deploying $NAME-inference (/v1/detect): autoscaled"
  write_env false
  local max=10; $GPU && max=4
  gcloud run deploy "$NAME-inference" "${common[@]}" --env-vars-file "$envs" \
    --min-instances 1 --max-instances "$max" --concurrency 4
  rm -f "$envs"
  local manager inference
  manager=$(gcloud run services describe "$NAME-manager" --region "$REGION" --format 'value(status.url)')
  inference=$(gcloud run services describe "$NAME-inference" --region "$REGION" --format 'value(status.url)')
  summary "### ForensiX on Cloud Run ($TARGET) is deployed" "" \
    "- Web manager: $manager/ui/" \
    "- Detection API (autoscaled): $inference/v1/detect" \
    "- API key: $(key_help "$API_SECRET")" \
    "- Admin key: $(key_help "$ADMIN_SECRET")" \
    "- Logs: \`gcloud run services logs tail $NAME-manager --region $REGION\`"
}

# -- GKE Autopilot: the Kustomize overlay, Filestore, GKE Ingress --------------------------------
gke() {
  need kubectl
  local overlay="$REPO/deploy/kubernetes/overlays/$TARGET"
  if $DESTROY; then
    log "Deleting ForensiX from cluster $NAME (load balancer and Filestore volume first), then the cluster"
    if gcloud container clusters get-credentials "$NAME" --region "$REGION" >/dev/null 2>&1; then
      kubectl delete -k "$overlay" --ignore-not-found --wait=true || true
      gcloud container clusters delete "$NAME" --region "$REGION" --quiet
    fi
    delete_keys
    summary "Deleted GKE cluster \`$NAME\` and its keys in $REGION ($PROJECT)."
    return
  fi
  log "Enabling APIs"
  gcloud services enable container.googleapis.com file.googleapis.com secretmanager.googleapis.com
  if ! gcloud container clusters describe "$NAME" --region "$REGION" >/dev/null 2>&1; then
    log "Creating GKE Autopilot cluster $NAME in $REGION (about 10 minutes)"
    gcloud container clusters create-auto "$NAME" --region "$REGION"
  fi
  gcloud container clusters get-credentials "$NAME" --region "$REGION"
  ensure_keys
  kubectl create namespace forensix --dry-run=client -o yaml | kubectl apply -f -
  local env_file; env_file=$(mktemp); chmod 600 "$env_file"
  {
    echo "FORENSIX_SERVER_API_KEY=$(gcloud secrets versions access latest --secret "$API_SECRET")"
    echo "FORENSIX_SERVER_ADMIN_KEY=$(gcloud secrets versions access latest --secret "$ADMIN_SECRET")"
  } > "$env_file"
  kubectl -n forensix create secret generic forensix-keys --from-env-file="$env_file" --dry-run=client -o yaml | kubectl apply -f -
  rm -f "$env_file"
  log "Deploying the $TARGET overlay"
  if [[ -n "$IMAGE" ]]; then
    kubectl kustomize "$overlay" | sed "s|image: ghcr.io/anas-rz/forensix:[a-z-]*|image: $IMAGE|" | kubectl apply -f -
  else
    kubectl apply -k "$overlay"
  fi
  log "Waiting for the pods (Autopilot adds nodes on demand; loading models takes minutes)"
  kubectl -n forensix rollout status deployment/forensix-manager --timeout=45m
  kubectl -n forensix rollout status deployment/forensix-inference --timeout=45m
  local ip=""
  for _ in $(seq 60); do
    ip=$(kubectl -n forensix get ingress forensix -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    [[ -n "$ip" ]] && break; sleep 10
  done
  summary "### ForensiX on GKE ($TARGET) is deployed" "" \
    "- Web manager: http://${ip:-<pending: kubectl -n forensix get ingress forensix>}/ui/ (the load balancer needs a few more minutes to pass health checks)" \
    "- API key: $(key_help "$API_SECRET")" \
    "- Admin key: $(key_help "$ADMIN_SECRET")" \
    "- Cluster: \`gcloud container clusters get-credentials $NAME --region $REGION\`, then \`kubectl -n forensix get pods\`"
}

case "$TARGET" in
  cloudrun|cloudrun-gpu) cloudrun ;;
  gke|gke-gpu) gke ;;
esac
