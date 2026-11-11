# Deploying on Azure

This guide is for engineers putting the ForensiX server (API and web manager) on Azure. Pick a
target, then deploy it with the "Deploy to Azure" button, with one command, or with one click in
GitHub Actions. Every deployment lives in its own resource group, so the teardown deletes that
group.

| Target | What runs | Pick it when |
|---|---|---|
| **`aca`**: Container Apps, CPU | Serverless containers on a dedicated D8 workload profile, HTTPS included | The simplest service, moderate volume, mostly images and audio |
| **`aca-gpu`**: Container Apps, GPU | Serverless NVIDIA T4 per replica | Video and audiovisual models without managing anything |
| **`aks`**: AKS | Kubernetes with the cluster autoscaler and managed NGINX ingress | You run Kubernetes already, or want the same manifests as on other clouds |
| **`aks-gpu`**: AKS, GPU | Kubernetes plus an autoscaled T4 node pool | Kubernetes plus GPUs |

| File | What it is |
|---|---|
| [`deploy/azure/containerapps.bicep`](../deploy/azure/containerapps.bicep) | Bicep template: the whole Container Apps deployment |
| [`deploy/azure/containerapps.json`](../deploy/azure/containerapps.json) | The same compiled to an ARM template, for the "Deploy to Azure" button |
| [`deploy/azure/deploy.sh`](../deploy/azure/deploy.sh) | One command to deploy or tear down any target |
| [`deploy/kubernetes/overlays/aks`](../deploy/kubernetes/overlays/aks/), [`aks-gpu`](../deploy/kubernetes/overlays/aks-gpu/) | The [Kubernetes manifests](cloud.md#kubernetes) adapted to AKS |
| [`.github/workflows/deploy-azure.yml`](../.github/workflows/deploy-azure.yml) | The same, as a one-click GitHub Actions workflow |

## Before you start: the image

Every target runs `ghcr.io/anas-rz/forensix:cpu-bundled` or `:gpu-bundled`, which have the
default models baked in. Push a version tag (`git tag v0.1.0 && git push origin v0.1.0`) to
publish them, then make the GHCR package public ([details](aws.md#before-you-start-the-image)).
To use a private registry (Azure Container Registry), pass `--image` and give the apps or cluster
pull access (`az containerapp registry set`, `az aks update --attach-acr`).

## Option 1: Deploy to Azure (Container Apps)

[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fanas-rz%2FForensiX%2Fmain%2Fdeploy%2Fazure%2Fcontainerapps.json)

1. Choose or create a resource group and a region. `compute = gpu` needs a serverless-GPU region
   such as `westus3`, `swedencentral` or `australiaeast`.
2. Enter an **API key** and an **admin key**: long random strings, for example from
   `python3 -c "import secrets; print(secrets.token_urlsafe(30))"`.
3. **Review + create.** When it finishes, the deployment's **Outputs** show the web manager and
   detection URLs.

The button deploys the JSON template from the `main` branch. After editing the Bicep file,
regenerate the JSON with
`az bicep build --file deploy/azure/containerapps.bicep --outfile deploy/azure/containerapps.json`.

## Option 2: one command

Needs the [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli), logged in
(`az login`) with a subscription selected. AKS also needs `kubectl` (`az aks install-cli`).

```bash
deploy/azure/deploy.sh aca                      # or aca-gpu, aks, aks-gpu
deploy/azure/deploy.sh aca --destroy            # delete the resource group
```

| Option | Default | Meaning |
|---|---|---|
| `--name` | `forensix` | Resource group and resource name prefix |
| `--location` | `AZURE_LOCATION`, else `eastus` | Region. `aca-gpu` needs a serverless-GPU region; `aks-gpu` needs NCasT4_v3 quota |
| `--image` | `ghcr.io/anas-rz/forensix:{cpu,gpu}-bundled` | Your image |

Rerunning the deploy updates in place and keeps the keys. The deploy generates them once,
stores them as Container Apps or Kubernetes secrets, and never prints them. The output shows how
to read them:

```bash
az containerapp secret show -g forensix -n forensix-manager --secret-name api-key --query value -o tsv
curl -H "X-API-Key: $KEY" -F file=@photo.jpg https://<inference URL>/v1/detect
```

## Option 3: one click in GitHub Actions

**Actions → deploy-azure → Run workflow**: pick the target, `deploy` or `destroy`, the name and the
region. The URLs and key commands appear in the run summary.

**One-time setup:** GitHub signs in to Microsoft Entra with OpenID Connect, so no client secret
is stored.

```bash
SUB=$(az account show --query id -o tsv)
APP=$(az ad app create --display-name forensix-deploy --query appId -o tsv)
az ad sp create --id $APP
az role assignment create --assignee $APP --role Owner --scope /subscriptions/$SUB
az ad app federated-credential create --id $APP --parameters '{
  "name": "github-forensix-azure",
  "issuer": "https://token.actions.githubusercontent.com",
  "subject": "repo:anas-rz/ForensiX:environment:azure",
  "audiences": ["api://AzureADTokenExchange"]
}'
```

A deploy creates resource groups, clusters and role assignments for managed identities, so the
app needs Owner, or Contributor plus User Access Administrator. In a shared tenant, scope it to a
dedicated subscription.

In the repository, create the environment `azure` (**Settings → Environments**; add required
reviewers to gate deploys). Then add these repository secrets:
- `AZURE_CLIENT_ID` = `$APP`
- `AZURE_TENANT_ID` = from `az account show --query tenantId -o tsv`
- `AZURE_SUBSCRIPTION_ID` = `$SUB`

## What gets deployed

**Container Apps** (`aca`, `aca-gpu`; `deploy/azure/containerapps.bicep`):

- **`<name>-manager`** serves the web manager, dataset uploads, calibration and fairness jobs. It
  runs as exactly one replica, the single writer of shared state.
- **`<name>-inference`** serves `/v1/detect`. It scales from 1 to 6 replicas on concurrent
  requests and can't write shared state.
- **Two URLs.** Each app has its own HTTPS URL on `azurecontainerapps.io`. Open the web manager
  on the manager URL and send API traffic to the inference URL. (The manager also answers
  `/v1/detect`, which the web manager's Test page uses.) To put both behind one domain, front
  them with Azure Front Door or Application Gateway and route the management paths (see the
  [Ingress](cloud.md#what-gets-deployed)) to the manager.
- **Compute:**
  - **CPU:** the dedicated `D8` workload profile, with one replica per node (7 vCPUs, 28 GiB).
    The Consumption profile tops out at 4 vCPUs and 8 GiB, too little for the default models.
  - **GPU:** the serverless `Consumption-GPU-NC8as-T4` profile, with one T4, 8 vCPUs and 56 GiB per
    replica. Serverless GPUs need a quota request.
- **Shared state:** an Azure Files share mounted at `/data` in both apps, owned by the image's
  user (uid 10001).
- **Probes:** startup and readiness on `/ready` (up to 15 minutes for models to load), liveness
  on `/health`.
- **Logs:** in a Log Analytics workspace.

**AKS** (`aks`, `aks-gpu`):

- **Cluster:**
  - two to six `Standard_D8s_v5` nodes, scaled by the cluster autoscaler;
  - Azure CNI overlay with the Cilium dataplane, which enforces the NetworkPolicy;
  - the application routing add-on, a managed NGINX ingress.
- **GPU:** `aks-gpu` adds a `gpu` node pool of `Standard_NC4as_T4_v3` (one to four nodes) and the
  NVIDIA device plugin. The pool has the `nvidia.com/gpu.present=true` label and
  `nvidia.com/gpu` taint the [GPU overlay](cloud.md#kubernetes) expects.
- **Workloads:** the `aks` or `aks-gpu` overlay, with the shared volume on Azure Files
  (`azurefile-csi`). The keys are in the `forensix-keys` Kubernetes Secret.

## HTTPS and a domain

- **Container Apps:** HTTPS is on by default. Add a custom domain with a free managed certificate
  using `az containerapp hostname add` and `az containerapp hostname bind`.
- **AKS:** the Ingress serves plain HTTP on its IP. Point a host at that IP and add a TLS
  certificate through the application routing add-on's Key Vault integration. Until then,
  restrict access or don't use real keys over it.

## Costs and teardown

- **Container Apps CPU:** at least one D8 node per running replica, so two at minimum (manager
  plus one inference replica).
- **Container Apps GPU:** billed per second while replicas run.
- **AKS:** the nodes (two or more D8s_v5, plus T4 nodes for GPU), the load balancer and the
  Azure Files share.

`--destroy` (or the workflow's `destroy` action) deletes the resource group, including the file
share with your datasets and calibrations.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Revision fails with an image pull error | The GHCR image isn't published or isn't public: see [the image](#before-you-start-the-image) |
| `aca-gpu`: workload profile not available or quota exceeded | Use a serverless-GPU region and request *Consumption-GPU-NC8as-T4* quota for Container Apps |
| `aks-gpu`: node pool creation fails | Request quota for the *Standard NCASv3_T4 Family vCPUs* in the region |
| Replicas restart while loading | Startup takes longer than the probe allows on a slow region; raise `periodSeconds` in the Bicep probes |
| `POST /v1/datasets` returns 409 | You sent it to the inference app; use the manager URL |
| Logs | `az containerapp logs show -g RG -n <name>-manager --follow`, or `kubectl -n forensix logs deploy/forensix-manager` |

## What's been verified

- The Bicep template compiles with `az bicep build`, with no linter warnings.
- The deploy script passes ShellCheck.
- The `aks` and `aks-gpu` overlay patches apply cleanly to the base manifests.

Deployments to an Azure subscription haven't been run from this repository yet. Try a
non-production subscription first.

**Next step:** open the web manager's [dashboard](ui.md#the-overview-dashboard), then [calibrate](calibration.md) on your media.
