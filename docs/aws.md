# Deploying on AWS

This guide is for engineers putting the ForensiX server (API and web manager) on AWS. Pick a
target, then deploy it in one of three ways: from the AWS console, with one command, or with
one click in GitHub Actions. Everything the deploy creates is removed by the matching teardown.

| Target | What runs | Pick it when |
|---|---|---|
| **`ecs`**: ECS on Fargate | CPU tasks, no servers to manage | You want the simplest service, moderate volume, mostly images and audio |
| **`ecs-gpu`**: ECS on EC2 GPU | One NVIDIA L4 (`g6.2xlarge`) per task | Video and audiovisual models at volume, still without Kubernetes |
| **`eks`**: EKS Auto Mode | Kubernetes, CPU nodes AWS manages and scales | You run Kubernetes already, or want the same manifests as on other clouds |
| **`eks-gpu`**: EKS Auto Mode, GPU | Kubernetes, NVIDIA nodes launched on demand | Kubernetes plus GPUs |

| File | What it is |
|---|---|
| [`deploy/aws/ecs.yaml`](../deploy/aws/ecs.yaml) | CloudFormation template: the whole ECS deployment in one stack |
| [`deploy/aws/eks/`](../deploy/aws/eks/) | eksctl cluster config (Auto Mode), ALB ingress class, GPU node pool, EFS storage class |
| [`deploy/kubernetes/overlays/eks`](../deploy/kubernetes/overlays/eks/), [`eks-gpu`](../deploy/kubernetes/overlays/eks-gpu/) | The [Kubernetes manifests](cloud.md#kubernetes) adapted to EKS |
| [`deploy/aws/deploy.sh`](../deploy/aws/deploy.sh) | One command to deploy or tear down any target |
| [`.github/workflows/deploy-aws.yml`](../.github/workflows/deploy-aws.yml) | The same, as a one-click GitHub Actions workflow |

## Before you start: the image

Every target runs the published image, `ghcr.io/anas-rz/forensix:cpu-bundled` or `:gpu-bundled`,
with the default models baked in. The [docker workflow](cloud.md#the-image) publishes them when
you push a version tag:

```bash
git tag v0.1.0 && git push origin v0.1.0     # builds and pushes cpu, gpu, cpu-bundled, gpu-bundled
```

GitHub creates the package as **private**. AWS can only pull it anonymously once you make it public
(GitHub → your profile → Packages → `forensix` → Package settings → Change visibility). To keep it
private, push the image to Amazon ECR instead and pass it with `--image` (or the `image` input):

```bash
aws ecr create-repository --repository-name forensix
aws ecr get-login-password | docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.REGION.amazonaws.com
docker build -t ACCOUNT.dkr.ecr.REGION.amazonaws.com/forensix:cpu-bundled \
  --build-arg MODELS="dfdet-clip commfor nes2net-x-itw syncnet c2pa watermark-image watermark-audio metadata" .
docker push ACCOUNT.dkr.ecr.REGION.amazonaws.com/forensix:cpu-bundled
```

## Option 1: from the AWS console (ECS)

No tools needed. In the [CloudFormation console](https://console.aws.amazon.com/cloudformation/home#/stacks/create):

1. **Create stack** → **Upload a template file** → choose `deploy/aws/ecs.yaml`.
2. Name the stack (e.g. `forensix`) and pick **Compute**: `Fargate` or `GPU`. The other
   parameters have working defaults.
3. Acknowledge that the stack creates IAM roles, then **Submit**.
4. When the stack reaches `CREATE_COMPLETE` (15–30 minutes), its **Outputs** tab has the web
   manager URL and the Secrets Manager names of the generated API and admin keys.

**A "Launch Stack" button.** To give others a one-click link, upload the template to an S3 bucket
and link to the quick-create page:

```bash
aws s3 cp deploy/aws/ecs.yaml s3://YOUR-BUCKET/forensix/ecs.yaml
# https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/quickcreate?stackName=forensix&templateURL=https://YOUR-BUCKET.s3.amazonaws.com/forensix/ecs.yaml
```

```markdown
[![Launch Stack](https://s3.amazonaws.com/cloudformation-examples/cloudformation-launch-stack.png)](https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/quickcreate?stackName=forensix&templateURL=https://YOUR-BUCKET.s3.amazonaws.com/forensix/ecs.yaml)
```

## Option 2: one command

Needs the AWS CLI v2 with credentials, and for EKS also
[eksctl](https://eksctl.io/installation/) and [kubectl](https://kubernetes.io/docs/tasks/tools/).

```bash
deploy/aws/deploy.sh ecs                     # or ecs-gpu, eks, eks-gpu
deploy/aws/deploy.sh ecs --destroy           # delete everything it created
```

| Option | Default | Meaning |
|---|---|---|
| `--name` | `forensix` | Stack or cluster name; deploy several side by side with different names |
| `--region` | `AWS_REGION`, else `us-east-1` | Region. GPU targets need L4/A10G capacity there |
| `--image` | `ghcr.io/anas-rz/forensix:{cpu,gpu}-bundled` | Your image, e.g. from ECR |
| `--certificate-arn` | none | ECS: an ACM certificate, to serve HTTPS (HTTP redirects to it) |
| `--admin-arn` | none | EKS: an IAM user or role to make cluster admin, e.g. yours when GitHub Actions created the cluster |

Rerunning the deploy updates in place: a new image, a changed parameter, or a fresh rollout.

When it finishes it prints the web manager URL and how to read the keys. The keys are generated
in AWS Secrets Manager and never printed:

```bash
aws secretsmanager get-secret-value --secret-id <name from the output> --query SecretString --output text
```

Enter them under **Settings** in the web manager, or send `X-API-Key` to the API:

```bash
curl -H "X-API-Key: $KEY" -F file=@photo.jpg http://<load balancer>/v1/detect
```

## Option 3: one click in GitHub Actions

**Actions → deploy-aws → Run workflow**: pick the target, `deploy` or `destroy`, the name and the
region. The workflow runs `deploy/aws/deploy.sh` and writes the URL and key commands to the run
summary.

**One-time setup:** GitHub signs in to AWS with OpenID Connect, so no access keys are stored.

1. Add GitHub as an identity provider in your AWS account (once per account):

    ```bash
    aws iam create-open-id-connect-provider --url https://token.actions.githubusercontent.com \
      --client-id-list sts.amazonaws.com
    ```

2. Create a role that only this repository's `aws` environment can assume. Save as `trust.json`,
   replacing `ACCOUNT`:

    ```json
    {
      "Version": "2012-10-17",
      "Statement": [{
        "Effect": "Allow",
        "Principal": {"Federated": "arn:aws:iam::ACCOUNT:oidc-provider/token.actions.githubusercontent.com"},
        "Action": "sts:AssumeRoleWithWebIdentity",
        "Condition": {"StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
          "token.actions.githubusercontent.com:sub": "repo:anas-rz/ForensiX:environment:aws"
        }}
      }]
    }
    ```

    ```bash
    aws iam create-role --role-name forensix-deploy --assume-role-policy-document file://trust.json
    aws iam attach-role-policy --role-name forensix-deploy --policy-arn arn:aws:iam::aws:policy/AdministratorAccess
    ```

    A deploy creates VPCs, IAM roles, load balancers, EFS, ECS or EKS, so the role needs broad
    rights. In a shared account, use a dedicated account or replace `AdministratorAccess` with a
    policy limited to CloudFormation, EC2, ECS, EKS, ELB, EFS, IAM, Auto Scaling, Secrets Manager,
    CloudWatch Logs and SSM.

3. In the repository: **Settings → Environments → New environment** `aws` (add required reviewers
   to approve each deploy), then **Settings → Secrets and variables → Actions → New repository
   secret** `AWS_DEPLOY_ROLE_ARN` = the role's ARN.

For EKS, pass your own IAM ARN as `admin_arn` so you can use `kubectl` on the cluster the workflow
created.

## What gets deployed

Both platforms deploy the same design as the [Kubernetes manifests](cloud.md#what-gets-deployed).

- **One manager** serves the web manager, dataset uploads, calibration and fairness jobs, and
  identity enrollment. It's the single writer of shared state.
- **Autoscaled inference replicas** serve `/v1/detect`. They can't write shared state.
- **EFS** holds the shared state: datasets, calibration versions, fairness reports and galleries.
  An activated calibration applies to every replica on its next request.
- **An Application Load Balancer** routes management paths to the manager and everything else to
  the replicas. It health-checks `/ready`, so a task gets traffic only once its models are loaded.

**ECS** (`deploy/aws/ecs.yaml`):

- **Network:** a new VPC with two public subnets. Tasks accept traffic only from the load
  balancer, and EFS only from the tasks.
- **Fargate:** 4 vCPUs, 16 GiB and 100 GiB of ephemeral storage per task. The inference service
  scales from 2 to 6 tasks on CPU utilization.
- **GPU:**
  - An Auto Scaling group of ECS GPU-optimized instances (`g6.2xlarge` by default, one task each),
    sized by an ECS capacity provider.
  - Inference scales on requests per task, because GPU work doesn't show up as CPU.
  - GPU tasks have no public IP. The instance pulls the image, and the bundled image needs no
    other downloads.
- **Keys and logs:** both keys are generated in Secrets Manager and injected as secrets. Logs go
  to CloudWatch (`/forensix/<stack>`, 30 days).
- **Parameters:** instance type, task counts, preloaded models, image, certificate and the allowed
  client CIDR. Set them in the console, or with `aws cloudformation deploy --parameter-overrides`.

**EKS** (`deploy.sh eks`):

- **Cluster:** an [Auto Mode](https://docs.aws.amazon.com/eks/latest/userguide/automode.html)
  cluster. AWS runs the nodes (Karpenter), the load balancer controller and the GPU drivers, so
  nodes appear when the HPA adds pods and go away when they're idle.
- **Storage:** an EFS file system with mount targets in the cluster's private subnets, plus the
  `efs-sc` storage class and the EFS CSI driver add-on.
- **Keys:** generated into the Secrets Manager secret `<name>/forensix-keys`, then copied into
  the `forensix-keys` Kubernetes Secret.
- **Workloads:** the `eks` or `eks-gpu` overlay. Its Ingress becomes an internet-facing ALB, and
  for `eks-gpu` a `gpu` NodePool (g6, g6e or g5, one GPU per node) is added.

## HTTPS and a domain

- **ECS:** pass `--certificate-arn` (an ACM certificate in the same region). Port 80 then
  redirects to 443. Point a CNAME or a Route 53 alias at the `LoadBalancerDns` output.
- **EKS:** add these annotations to the Ingress in your copy of the overlay, then point your DNS
  at the Ingress address:
  - `alb.ingress.kubernetes.io/certificate-arn: <arn>`
  - `alb.ingress.kubernetes.io/listen-ports: '[{"HTTP": 80}, {"HTTPS": 443}]'`
  - `alb.ingress.kubernetes.io/ssl-redirect: "443"`

Until then, traffic, including API keys, is plain HTTP. Restrict who can reach it (ECS
`AllowedCidr`, or `alb.ingress.kubernetes.io/inbound-cidrs` on EKS) or add HTTPS before using
real keys.

## Costs and teardown

You pay for what runs:
- **Fixed:** the load balancer and the EFS storage.
- **Per task or node:** Fargate tasks or EC2/EKS nodes. GPU instances are the largest cost.
- **EKS only:** the control plane and a NAT gateway.

`--destroy` (or the workflow's `destroy` action) deletes the stack or cluster, the EFS file
system with your datasets and calibrations, and the keys. Copy anything you want to keep first,
for example by downloading calibration tables from the web manager.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Tasks or pods stop with `CannotPullContainerError` / `ImagePullBackOff` | The image isn't published or isn't public: see [the image](#before-you-start-the-image) |
| Load balancer returns 503 for a few minutes after deploy | Models are loading; targets join once `/ready` returns 200 |
| ECS GPU stack waits on the services | No GPU capacity in the region or AZs, or an EC2 vCPU quota for G instances: request a quota increase or try another region |
| EKS GPU pods stay `Pending` | Same capacity or quota limit; `kubectl describe nodeclaims` shows why nodes didn't launch |
| `POST /v1/datasets` or calibration returns 409 | The request reached an inference replica; use the load balancer's URL, which routes management paths to the manager |
| Logs | ECS: `aws logs tail --follow /forensix/<name>`. EKS: `kubectl -n forensix logs deploy/forensix-manager` |

## What's been verified

- The CloudFormation template passes `cfn-lint`.
- The deploy script passes ShellCheck.
- The eksctl config validates against eksctl's schema.
- The EKS overlay patches apply cleanly to the base manifests.

Deployments to an AWS account haven't been run from this repository yet. Try a non-production
account first, and report what needed changing.
