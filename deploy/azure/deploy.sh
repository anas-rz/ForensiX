#!/usr/bin/env bash
# Deploy ForensiX to Azure, or tear it down, with one command. See docs/azure.md.
#
#   deploy/azure/deploy.sh aca               # Azure Container Apps, CPU (dedicated D8 profile)
#   deploy/azure/deploy.sh aca-gpu           # Azure Container Apps, serverless NVIDIA T4
#   deploy/azure/deploy.sh aks               # AKS cluster, CPU nodes with autoscaling
#   deploy/azure/deploy.sh aks-gpu           # AKS cluster plus an NVIDIA T4 node pool
#   deploy/azure/deploy.sh aca --destroy     # delete the resource group and everything in it
#
# Options: --name NAME (default forensix; also the resource group), --location LOCATION (default:
#          AZURE_LOCATION, then eastus; aca-gpu needs a serverless-GPU region such as westus3),
#          --image IMAGE (default ghcr.io/anas-rz/forensix:{cpu,gpu}-bundled).
# Needs: az CLI (logged in, subscription selected); for AKS also kubectl.
set -euo pipefail

usage() { sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

TARGET="${1:-}"; [[ "$TARGET" =~ ^(aca|aca-gpu|aks|aks-gpu)$ ]] || usage 1
shift
NAME=forensix LOCATION="${AZURE_LOCATION:-eastus}" IMAGE="" DESTROY=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --location) LOCATION="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --destroy) DESTROY=true; shift ;;
    -h|--help) usage ;;
    *) echo "unknown option: $1" >&2; usage 1 ;;
  esac
done
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
RG="$NAME"
log() { printf '\n==> %s\n' "$*"; }
need() { command -v "$1" >/dev/null || { echo "$1 is required (see docs/azure.md)" >&2; exit 1; }; }
summary() { [[ -n "${GITHUB_STEP_SUMMARY:-}" ]] && printf '%s\n' "$@" >> "$GITHUB_STEP_SUMMARY"; printf '%s\n' "$@"; }
new_key() { python3 -c 'import secrets; print(secrets.token_urlsafe(30))'; }

need az
az account show --query id -o tsv >/dev/null
GPU=false; [[ "$TARGET" == *-gpu ]] && GPU=true

if $DESTROY; then
  log "Deleting resource group $RG and everything in it (datasets and calibrations included)"
  az group delete --name "$RG" --yes
  summary "Deleted ForensiX resource group \`$RG\`."
  exit 0
fi
az group create --name "$RG" --location "$LOCATION" --output none

# -- Container Apps: one Bicep deployment -------------------------------------------------------
aca() {
  local compute=cpu; $GPU && compute=gpu
  log "Registering resource providers"
  for ns in Microsoft.App Microsoft.OperationalInsights Microsoft.Storage; do az provider register --namespace "$ns" --wait --output none; done
  # Keys are generated once and kept as Container Apps secrets: reuse them on redeploys.
  local api admin
  api=$(az containerapp secret show -g "$RG" -n "$NAME-manager" --secret-name api-key --query value -o tsv 2>/dev/null || true)
  admin=$(az containerapp secret show -g "$RG" -n "$NAME-manager" --secret-name admin-key --query value -o tsv 2>/dev/null || true)
  [[ -n "$api" ]] || api=$(new_key)
  [[ -n "$admin" ]] || admin=$(new_key)
  local params; params=$(mktemp); chmod 600 "$params"
  python3 - "$params" "$NAME" "$compute" "$IMAGE" "$api" "$admin" <<'EOF'
import json, sys
path, name, compute, image, api, admin = sys.argv[1:]
values = {"name": name, "compute": compute, "image": image, "apiKey": api, "adminKey": admin}
json.dump({"$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentParameters.json#",
           "contentVersion": "1.0.0.0", "parameters": {k: {"value": v} for k, v in values.items()}}, open(path, "w"))
EOF
  log "Deploying Container Apps ($compute) to $RG in $LOCATION; the first deploy takes 10-20 minutes"
  az deployment group create -g "$RG" --name forensix --template-file "$HERE/containerapps.bicep" \
    --parameters "@$params" --output none || { rm -f "$params"; exit 1; }
  rm -f "$params"
  local manager detect
  out() { az deployment group show -g "$RG" --name forensix --query "properties.outputs.$1.value" -o tsv; }
  manager=$(out managerUrl); detect=$(out detectUrl)
  summary "### ForensiX on Azure Container Apps ($compute) is deployed" "" \
    "- Web manager: $manager (models finish loading a few minutes after the deploy)" \
    "- Detection API (autoscaled): $detect" \
    "- API key: \`az containerapp secret show -g $RG -n $NAME-manager --secret-name api-key --query value -o tsv\`" \
    "- Admin key: the same with \`--secret-name admin-key\`" \
    "- Logs: \`az containerapp logs show -g $RG -n $NAME-manager --follow\`"
}

# -- AKS: cluster with autoscaling, app routing (managed NGINX), the Kustomize overlay ----------
aks() {
  need kubectl
  local overlay="$REPO/deploy/kubernetes/overlays/$TARGET"
  if ! az aks show -g "$RG" -n "$NAME" --output none 2>/dev/null; then
    log "Creating AKS cluster $NAME in $LOCATION (about 10 minutes)"
    az aks create -g "$RG" -n "$NAME" --location "$LOCATION" \
      --node-vm-size Standard_D8s_v5 --node-count 2 --enable-cluster-autoscaler --min-count 2 --max-count 6 \
      --network-plugin azure --network-plugin-mode overlay --network-dataplane cilium \
      --enable-app-routing --generate-ssh-keys --output none
  fi
  if $GPU && ! az aks nodepool show -g "$RG" --cluster-name "$NAME" -n gpu --output none 2>/dev/null; then
    log "Adding a T4 GPU node pool (label and taint match the gpu overlays)"
    az aks nodepool add -g "$RG" --cluster-name "$NAME" -n gpu --node-vm-size Standard_NC4as_T4_v3 \
      --enable-cluster-autoscaler --min-count 1 --max-count 4 \
      --labels nvidia.com/gpu.present=true --node-taints nvidia.com/gpu=present:NoSchedule --output none
  fi
  az aks get-credentials -g "$RG" -n "$NAME" --overwrite-existing
  if $GPU; then
    # AKS installs the NVIDIA driver on GPU nodes; the device plugin advertises nvidia.com/gpu to the scheduler
    kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml
  fi
  kubectl create namespace forensix --dry-run=client -o yaml | kubectl apply -f -
  if ! kubectl -n forensix get secret forensix-keys >/dev/null 2>&1; then
    local env_file; env_file=$(mktemp); chmod 600 "$env_file"
    printf 'FORENSIX_SERVER_API_KEY=%s\nFORENSIX_SERVER_ADMIN_KEY=%s\n' "$(new_key)" "$(new_key)" > "$env_file"
    kubectl -n forensix create secret generic forensix-keys --from-env-file="$env_file"
    rm -f "$env_file"
  fi
  log "Deploying the $TARGET overlay"
  if [[ -n "$IMAGE" ]]; then
    kubectl kustomize "$overlay" | sed "s|image: ghcr.io/anas-rz/forensix:[a-z-]*|image: $IMAGE|" | kubectl apply -f -
  else
    kubectl apply -k "$overlay"
  fi
  log "Waiting for the pods (loading models takes minutes)"
  kubectl -n forensix rollout status deployment/forensix-manager --timeout=45m
  kubectl -n forensix rollout status deployment/forensix-inference --timeout=45m
  local ip=""
  for _ in $(seq 60); do
    ip=$(kubectl -n forensix get ingress forensix -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    [[ -n "$ip" ]] && break; sleep 10
  done
  summary "### ForensiX on AKS ($TARGET) is deployed" "" \
    "- Web manager: http://${ip:-<pending: kubectl -n forensix get ingress forensix>}/ui/" \
    "- API key: \`kubectl -n forensix get secret forensix-keys -o jsonpath='{.data.FORENSIX_SERVER_API_KEY}' | base64 -d\`" \
    "- Admin key: the same with \`FORENSIX_SERVER_ADMIN_KEY\`" \
    "- Cluster: \`az aks get-credentials -g $RG -n $NAME\`, then \`kubectl -n forensix get pods\`"
}

case "$TARGET" in
  aca|aca-gpu) aca ;;
  aks|aks-gpu) aks ;;
esac
