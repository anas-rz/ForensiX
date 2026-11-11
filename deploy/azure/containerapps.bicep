// ForensiX on Azure Container Apps: a manager (web UI, datasets, calibration) and autoscaled
// inference apps, sharing state on an Azure Files share. CPU runs on a dedicated D8 workload
// profile; GPU on serverless NVIDIA T4. Deployed by deploy/azure/deploy.sh (see docs/azure.md).

@description('Prefix for every resource')
param name string = 'forensix'
param location string = resourceGroup().location
@allowed(['cpu', 'gpu'])
param compute string = 'cpu'
@description('Image repository and tag; empty = ghcr.io/anas-rz/forensix:{cpu,gpu}-bundled')
param image string = ''
param preload string = 'dfdet-clip commfor nes2net-x-itw syncnet c2pa watermark-image watermark-audio metadata'
@minValue(1)
param inferenceMinReplicas int = 1
@minValue(1)
param inferenceMaxReplicas int = 6
@secure()
@description('X-API-Key for /v1/detect and reads (deploy.sh generates it once and reuses it)')
param apiKey string
@secure()
@description('Admin key: datasets, calibration, model loading')
param adminKey string

var gpu = compute == 'gpu'
var containerImage = empty(image) ? 'ghcr.io/anas-rz/forensix:${gpu ? 'gpu-bundled' : 'cpu-bundled'}' : image
// D8: 8 vCPUs, 32 GiB per node, one replica each. NC8as-T4: 8 vCPUs, 56 GiB and one T4 per replica.
var gpuProfile = { name: 'gpu', workloadProfileType: 'Consumption-GPU-NC8as-T4' }
var cpuProfile = { name: 'compute', workloadProfileType: 'D8', minimumCount: 1, maximumCount: inferenceMaxReplicas + 1 }
var profile = gpu ? gpuProfile : cpuProfile
var resources = gpu ? { cpu: json('8'), memory: '56Gi' } : { cpu: json('7'), memory: '28Gi' }

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${name}-logs'
  location: location
  properties: { sku: { name: 'PerGB2018' }, retentionInDays: 30 }
}

// -- shared state: datasets, calibration versions, fairness reports, identity galleries -----------
resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: take('${replace(name, '-', '')}${uniqueString(resourceGroup().id)}', 24)
  location: location
  kind: 'StorageV2'
  sku: { name: 'Standard_LRS' }
  properties: { minimumTlsVersion: 'TLS1_2', allowBlobPublicAccess: false, supportsHttpsTrafficOnly: true }
}
resource fileService 'Microsoft.Storage/storageAccounts/fileServices@2023-05-01' = {
  parent: storage
  name: 'default'
}
resource share 'Microsoft.Storage/storageAccounts/fileServices/shares@2023-05-01' = {
  parent: fileService
  name: 'forensix-data'
  properties: { shareQuota: 1024 }
}

resource environment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${name}-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
    workloadProfiles: [
      { name: 'Consumption', workloadProfileType: 'Consumption' }
      profile
    ]
  }
}
resource environmentStorage 'Microsoft.App/managedEnvironments/storages@2024-03-01' = {
  parent: environment
  name: 'data'
  properties: {
    azureFile: {
      accountName: storage.name
      accountKey: storage.listKeys().keys[0].value
      shareName: share.name
      accessMode: 'ReadWrite'
    }
  }
}

// -- the two apps -----------------------------------------------------------------------------------
var secrets = [
  { name: 'api-key', value: apiKey }
  { name: 'admin-key', value: adminKey }
]
var volumes = [
  {
    name: 'data'
    storageType: 'AzureFile'
    storageName: environmentStorage.name
    mountOptions: 'uid=10001,gid=10001,dir_mode=0750,file_mode=0640'   // the image runs as uid 10001
  }
]
var probes = [
  // models load for minutes (longer on CPU): up to 15 minutes before the first /ready
  { type: 'Startup', httpGet: { path: '/ready', port: 8000 }, periodSeconds: 90, failureThreshold: 10, timeoutSeconds: 5 }
  { type: 'Readiness', httpGet: { path: '/ready', port: 8000 }, periodSeconds: 10, timeoutSeconds: 5 }
  { type: 'Liveness', httpGet: { path: '/health', port: 8000 }, periodSeconds: 20, timeoutSeconds: 5, failureThreshold: 3 }
]
func environmentVariables(device string, preload string, admin string) array => [
  { name: 'FORENSIX_SERVER_DEVICE', value: device }
  { name: 'FORENSIX_SERVER_PRELOAD', value: preload }
  { name: 'FORENSIX_SERVER_CALIBRATION_DIR', value: '/data/calibration' }
  { name: 'FORENSIX_SERVER_GALLERY_DIR', value: '/data/galleries' }
  { name: 'FORENSIX_SERVER_ADMIN_ENDPOINTS', value: admin }
  { name: 'FORENSIX_SERVER_UI', value: admin }
  { name: 'FORENSIX_SERVER_API_KEY', secretRef: 'api-key' }
  { name: 'FORENSIX_SERVER_ADMIN_KEY', secretRef: 'admin-key' }
  { name: 'LOG_FORMAT', value: 'json' }
]

// One replica: the single writer of shared state (web UI, uploads, calibration and fairness jobs).
resource manager 'Microsoft.App/containerApps@2024-03-01' = {
  name: '${name}-manager'
  location: location
  properties: {
    environmentId: environment.id
    workloadProfileName: profile.name
    configuration: {
      activeRevisionsMode: 'Single'
      secrets: secrets
      ingress: { external: true, targetPort: 8000, transport: 'auto', allowInsecure: false }
    }
    template: {
      containers: [
        {
          name: 'forensix'
          image: containerImage
          resources: resources
          env: environmentVariables(gpu ? 'cuda' : 'cpu', preload, 'true')
          volumeMounts: [{ volumeName: 'data', mountPath: '/data' }]
          probes: probes
        }
      ]
      volumes: volumes
      scale: { minReplicas: 1, maxReplicas: 1 }
    }
  }
}

// Prediction only (no writes to shared state, no UI), scaled on concurrent requests.
resource inference 'Microsoft.App/containerApps@2024-03-01' = {
  name: '${name}-inference'
  location: location
  properties: {
    environmentId: environment.id
    workloadProfileName: profile.name
    configuration: {
      activeRevisionsMode: 'Single'
      secrets: secrets
      ingress: { external: true, targetPort: 8000, transport: 'auto', allowInsecure: false }
    }
    template: {
      containers: [
        {
          name: 'forensix'
          image: containerImage
          resources: resources
          env: environmentVariables(gpu ? 'cuda' : 'cpu', preload, 'false')
          volumeMounts: [{ volumeName: 'data', mountPath: '/data' }]
          probes: probes
        }
      ]
      volumes: volumes
      scale: {
        minReplicas: inferenceMinReplicas
        maxReplicas: inferenceMaxReplicas
        rules: [{ name: 'concurrent-requests', http: { metadata: { concurrentRequests: '4' } } }]
      }
    }
  }
}

output managerUrl string = 'https://${manager.properties.configuration.ingress.fqdn}/ui/'
output detectUrl string = 'https://${inference.properties.configuration.ingress.fqdn}/v1/detect'
output logAnalyticsWorkspace string = logs.name
