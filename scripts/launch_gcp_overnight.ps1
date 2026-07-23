param(
    [string]$Project = "project-2696835e-1819-4c22-9e3",
    [string]$Zone = "us-central1-b",
    [string]$Bucket = "gs://project-2696835e-1819-4c22-9e3-ubx-overnight",
    [string]$OutputPrefix = "",
    [int]$TrainSteps = 1200,
    [int]$MaxSamples = 12000
)

$ErrorActionPreference = "Stop"
$gcloud = Join-Path $env:LOCALAPPDATA "Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
if (-not (Test-Path -LiteralPath $gcloud)) {
    throw "Google Cloud CLI is not installed."
}

$projectInfo = (& $gcloud compute project-info describe --project=$Project --format=json) | ConvertFrom-Json
$globalGpu = ($projectInfo.quotas | Where-Object { $_.metric -eq "GPUS_ALL_REGIONS" }).limit
if ([double]$globalGpu -lt 1) {
    throw "Global GPU quota is still $globalGpu. Wait for the submitted 0-to-1 quota request to be approved."
}

$globalCpu = ($projectInfo.quotas | Where-Object { $_.metric -eq "CPUS_ALL_REGIONS" }).limit
if ([double]$globalCpu -lt 48) {
    throw "Global CPU quota is still $globalCpu. Wait for the submitted 32-to-48 quota request to be approved."
}

$stamp = Get-Date -Format "yyyyMMdd-HHmm"
$instance = "ubx-fara-$stamp"
if (-not $OutputPrefix) {
    $OutputPrefix = "runs/$instance"
}
$metadata = @(
    "gcs-output=$Bucket/$OutputPrefix",
    "train-steps=$TrainSteps",
    "max-samples=$MaxSamples",
    "deadline-seconds=27000"
) -join ","

& $gcloud compute instances create $instance `
    --project=$Project `
    --zone=$Zone `
    --machine-type=g4-standard-48 `
    --provisioning-model=SPOT `
    --instance-termination-action=DELETE `
    --max-run-duration=8h `
    --maintenance-policy=TERMINATE `
    --no-restart-on-failure `
    --boot-disk-size=300GB `
    --boot-disk-type=hyperdisk-balanced `
    --image-family=common-cu129-ubuntu-2404-nvidia-580 `
    --image-project=deeplearning-platform-release `
    --service-account=92707153471-compute@developer.gserviceaccount.com `
    --scopes=https://www.googleapis.com/auth/cloud-platform `
    --metadata=$metadata `
    --metadata-from-file=startup-script=scripts/gcp_overnight_fara.sh

if ($LASTEXITCODE -ne 0) {
    throw "G4 creation failed. No training VM was launched."
}

& $gcloud compute instances describe $instance `
    --project=$Project `
    --zone=$Zone `
    --format="table(name,status,machineType.basename(),scheduling.provisioningModel,scheduling.maxRunDuration)"

Write-Output "INSTANCE=$instance"
Write-Output "LOGS: gcloud compute ssh $instance --zone=$Zone --command='sudo journalctl -u google-startup-scripts.service -f'"
Write-Output "CHECKPOINTS: $Bucket/$OutputPrefix"
