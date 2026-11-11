#!/usr/bin/env bash
# Deploy ForensiX to AWS, or tear it down, with one command. See docs/aws.md.
#
#   deploy/aws/deploy.sh ecs                 # ECS on Fargate (CPU), one CloudFormation stack
#   deploy/aws/deploy.sh ecs-gpu             # ECS on EC2 GPU instances
#   deploy/aws/deploy.sh eks                 # EKS Auto Mode cluster, CPU nodes
#   deploy/aws/deploy.sh eks-gpu             # EKS Auto Mode cluster, NVIDIA GPU nodes
#   deploy/aws/deploy.sh ecs --destroy       # delete everything the deploy created
#
# Options: --name NAME (default forensix), --region REGION (default: AWS_REGION, then us-east-1),
#          --image REPO:TAG (default ghcr.io/anas-rz/forensix:{cpu,gpu}-bundled),
#          --certificate-arn ARN (ECS: HTTPS), --admin-arn ARN (EKS: an IAM user/role given cluster admin).
# Needs: aws CLI v2 with credentials; for EKS also eksctl and kubectl.
set -euo pipefail

usage() { sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

TARGET="${1:-}"; [[ "$TARGET" =~ ^(ecs|ecs-gpu|eks|eks-gpu)$ ]] || usage 1
shift
NAME=forensix REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}" IMAGE="" CERT="" ADMIN_ARN="" DESTROY=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --certificate-arn) CERT="$2"; shift 2 ;;
    --admin-arn) ADMIN_ARN="$2"; shift 2 ;;
    --destroy) DESTROY=true; shift ;;
    -h|--help) usage ;;
    *) echo "unknown option: $1" >&2; usage 1 ;;
  esac
done
export AWS_REGION="$REGION" AWS_DEFAULT_REGION="$REGION"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
log() { printf '\n==> %s\n' "$*"; }
need() { command -v "$1" >/dev/null || { echo "$1 is required (see docs/aws.md)" >&2; exit 1; }; }
summary() { [[ -n "${GITHUB_STEP_SUMMARY:-}" ]] && printf '%s\n' "$@" >> "$GITHUB_STEP_SUMMARY"; printf '%s\n' "$@"; }

need aws
aws sts get-caller-identity --query Arn --output text >/dev/null

# -- ECS: one CloudFormation stack ---------------------------------------------------------------
ecs() {
  local compute=Fargate; [[ "$TARGET" == ecs-gpu ]] && compute=GPU
  if $DESTROY; then
    log "Deleting stack $NAME in $REGION (EFS data included)"
    aws cloudformation delete-stack --stack-name "$NAME"
    aws cloudformation wait stack-delete-complete --stack-name "$NAME"
    summary "Deleted ECS stack \`$NAME\` in $REGION."
    return
  fi
  local params=("Compute=$compute")
  if [[ -n "$IMAGE" ]]; then params+=("ImageRepository=${IMAGE%:*}" "ImageTag=${IMAGE##*:}"); fi
  if [[ -n "$CERT" ]]; then params+=("CertificateArn=$CERT"); fi
  log "Deploying stack $NAME ($compute) in $REGION; the first deploy takes 15-30 minutes"
  aws cloudformation deploy --stack-name "$NAME" --template-file "$HERE/ecs.yaml" \
    --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset --parameter-overrides "${params[@]}"
  local url api_secret admin_secret
  out() { aws cloudformation describe-stacks --stack-name "$NAME" --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text; }
  url=$(out Url); api_secret=$(out ApiKeySecret); admin_secret=$(out AdminKeySecret)
  summary "### ForensiX on ECS ($compute) is deployed" "" \
    "- Web manager: $url (models finish loading a few minutes after the stack completes)" \
    "- API key: \`aws secretsmanager get-secret-value --region $REGION --secret-id $api_secret --query SecretString --output text\`" \
    "- Admin key: \`aws secretsmanager get-secret-value --region $REGION --secret-id $admin_secret --query SecretString --output text\`" \
    "- Logs: \`aws logs tail --follow --region $REGION $(out LogGroup)\`"
}

# -- EKS: Auto Mode cluster, EFS, keys, the Kustomize overlay ------------------------------------
eks() {
  need eksctl; need kubectl
  local overlay="$REPO/deploy/kubernetes/overlays/$TARGET" secret="$NAME/forensix-keys"
  if $DESTROY; then eks_destroy "$overlay" "$secret"; return; fi

  if eksctl get cluster --name "$NAME" --region "$REGION" >/dev/null 2>&1; then
    log "Cluster $NAME exists; updating kubeconfig"
    aws eks update-kubeconfig --name "$NAME" --region "$REGION" >/dev/null
  else
    log "Creating EKS Auto Mode cluster $NAME in $REGION (about 15 minutes)"
    sed -e "s/__NAME__/$NAME/" -e "s/__REGION__/$REGION/" "$HERE/eks/cluster.yaml" | eksctl create cluster -f -
  fi
  if [[ -n "$ADMIN_ARN" ]]; then
    log "Giving $ADMIN_ARN cluster-admin access"
    aws eks create-access-entry --cluster-name "$NAME" --principal-arn "$ADMIN_ARN" >/dev/null 2>&1 || true
    aws eks associate-access-policy --cluster-name "$NAME" --principal-arn "$ADMIN_ARN" \
      --policy-arn arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy --access-scope type=cluster >/dev/null
  fi

  log "Shared storage: EFS in the cluster's VPC"
  local vpc cidr fs sg
  vpc=$(aws eks describe-cluster --name "$NAME" --query cluster.resourcesVpcConfig.vpcId --output text)
  cidr=$(aws ec2 describe-vpcs --vpc-ids "$vpc" --query 'Vpcs[0].CidrBlock' --output text)
  fs=$(aws efs describe-file-systems --creation-token "$NAME-data" --query 'FileSystems[0].FileSystemId' --output text)
  if [[ "$fs" == None ]]; then
    fs=$(aws efs create-file-system --creation-token "$NAME-data" --encrypted --throughput-mode elastic \
      --tags "Key=Name,Value=$NAME-data" --query FileSystemId --output text)
  fi
  until [[ $(aws efs describe-file-systems --file-system-id "$fs" --query 'FileSystems[0].LifeCycleState' --output text) == available ]]; do sleep 5; done
  sg=$(aws ec2 describe-security-groups --filters "Name=vpc-id,Values=$vpc" "Name=group-name,Values=$NAME-efs" \
    --query 'SecurityGroups[0].GroupId' --output text)
  if [[ "$sg" == None ]]; then
    sg=$(aws ec2 create-security-group --group-name "$NAME-efs" --description "ForensiX EFS (NFS from the cluster VPC)" \
      --vpc-id "$vpc" --query GroupId --output text)
    aws ec2 authorize-security-group-ingress --group-id "$sg" --protocol tcp --port 2049 --cidr "$cidr" >/dev/null
  fi
  for subnet in $(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$vpc" "Name=tag:kubernetes.io/role/internal-elb,Values=1" \
                    --query 'Subnets[].SubnetId' --output text); do
    aws efs create-mount-target --file-system-id "$fs" --subnet-id "$subnet" --security-groups "$sg" >/dev/null 2>&1 || true
  done
  sed "s/__FILE_SYSTEM_ID__/$fs/" "$HERE/eks/storageclass.yaml" | kubectl apply -f -
  kubectl apply -f "$HERE/eks/ingress-class.yaml"
  if [[ "$TARGET" == eks-gpu ]]; then kubectl apply -f "$HERE/eks/gpu-nodepool.yaml"; fi

  log "Keys: generated once and kept in Secrets Manager ($secret)"
  if ! aws secretsmanager describe-secret --secret-id "$secret" >/dev/null 2>&1; then
    local api admin
    api=$(aws secretsmanager get-random-password --exclude-punctuation --password-length 40 --query RandomPassword --output text)
    admin=$(aws secretsmanager get-random-password --exclude-punctuation --password-length 40 --query RandomPassword --output text)
    aws secretsmanager create-secret --name "$secret" --description "ForensiX API and admin keys ($NAME)" \
      --secret-string "{\"FORENSIX_SERVER_API_KEY\":\"$api\",\"FORENSIX_SERVER_ADMIN_KEY\":\"$admin\"}" >/dev/null
  fi
  kubectl create namespace forensix --dry-run=client -o yaml | kubectl apply -f -
  local env_file; env_file=$(mktemp); chmod 600 "$env_file"
  aws secretsmanager get-secret-value --secret-id "$secret" --query SecretString --output text |
    python3 -c 'import json, sys; print("\n".join(f"{k}={v}" for k, v in json.load(sys.stdin).items()))' > "$env_file"
  kubectl -n forensix create secret generic forensix-keys --from-env-file="$env_file" --dry-run=client -o yaml | kubectl apply -f -
  rm -f "$env_file"

  log "Deploying the $TARGET overlay"
  if [[ -n "$IMAGE" ]]; then
    kubectl kustomize "$overlay" | sed "s|image: ghcr.io/anas-rz/forensix:[a-z-]*|image: $IMAGE|" | kubectl apply -f -
  else
    kubectl apply -k "$overlay"
  fi
  log "Waiting for the pods (nodes are launched on demand; loading models takes minutes)"
  kubectl -n forensix rollout status deployment/forensix-manager --timeout=45m
  kubectl -n forensix rollout status deployment/forensix-inference --timeout=45m
  local host=""
  for _ in $(seq 60); do
    host=$(kubectl -n forensix get ingress forensix -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    [[ -n "$host" ]] && break; sleep 10
  done
  summary "### ForensiX on EKS ($TARGET) is deployed" "" \
    "- Web manager: http://${host:-<pending: kubectl -n forensix get ingress forensix>}/ui/ (the load balancer's DNS takes a few minutes to resolve)" \
    "- Keys: \`aws secretsmanager get-secret-value --region $REGION --secret-id $secret --query SecretString --output text\`" \
    "- Cluster: \`aws eks update-kubeconfig --region $REGION --name $NAME\`, then \`kubectl -n forensix get pods\`"
}

eks_destroy() {
  local overlay="$1" secret="$2" vpc fs sg
  log "Deleting ForensiX from cluster $NAME (removes the load balancer and volumes first)"
  if aws eks update-kubeconfig --name "$NAME" --region "$REGION" >/dev/null 2>&1; then
    kubectl delete -k "$overlay" --ignore-not-found --wait=true || true
    kubectl delete -f "$HERE/eks/gpu-nodepool.yaml" --ignore-not-found || true
    vpc=$(aws eks describe-cluster --name "$NAME" --query cluster.resourcesVpcConfig.vpcId --output text)
  fi
  fs=$(aws efs describe-file-systems --creation-token "$NAME-data" --query 'FileSystems[0].FileSystemId' --output text)
  if [[ "$fs" != None ]]; then
    log "Deleting EFS $fs"
    for mt in $(aws efs describe-mount-targets --file-system-id "$fs" --query 'MountTargets[].MountTargetId' --output text); do
      aws efs delete-mount-target --mount-target-id "$mt"
    done
    until [[ $(aws efs describe-mount-targets --file-system-id "$fs" --query 'length(MountTargets)' --output text) == 0 ]]; do sleep 10; done
    aws efs delete-file-system --file-system-id "$fs"
  fi
  if [[ -n "${vpc:-}" ]]; then
    sg=$(aws ec2 describe-security-groups --filters "Name=vpc-id,Values=$vpc" "Name=group-name,Values=$NAME-efs" \
      --query 'SecurityGroups[0].GroupId' --output text)
    [[ "$sg" != None ]] && aws ec2 delete-security-group --group-id "$sg"
  fi
  log "Deleting cluster $NAME (about 10 minutes)"
  eksctl delete cluster --name "$NAME" --region "$REGION" --wait || true
  aws secretsmanager delete-secret --secret-id "$secret" --force-delete-without-recovery >/dev/null 2>&1 || true
  summary "Deleted EKS cluster \`$NAME\`, its EFS file system and keys in $REGION."
}

case "$TARGET" in
  ecs|ecs-gpu) ecs ;;
  eks|eks-gpu) eks ;;
esac
