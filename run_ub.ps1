$ErrorActionPreference = "Stop"

$workspace = $PSScriptRoot
$python = Join-Path $workspace ".venv\Scripts\python.exe"
$pythonw = Join-Path $workspace ".venv\Scripts\pythonw.exe"
$worker = Join-Path $workspace "ub_worker.py"
$viewer = Join-Path $workspace "devtools\ub_viewer.py"

$game = "ls20"
$noViewer = $false
$workerArgs = @()
$launcherArgs = @($args)
$gameWasSet = $false

for ($index = 0; $index -lt $launcherArgs.Count; $index++) {
    $argument = [string]$launcherArgs[$index]

    if ($argument -in @("-Game", "--game")) {
        if ($index + 1 -ge $launcherArgs.Count) {
            throw "$argument requires a game id."
        }
        $index++
        $game = [string]$launcherArgs[$index]
        $gameWasSet = $true
        continue
    }

    if ($argument -match "^(?:-Game|--game)=(.+)$") {
        $game = $Matches[1]
        $gameWasSet = $true
        continue
    }

    if ($argument -in @("-NoViewer", "--no-viewer")) {
        $noViewer = $true
        continue
    }

    if ($argument -eq "--competition") {
        $noViewer = $true
        $workerArgs += $argument
        continue
    }

    if ($index -eq 0 -and -not $gameWasSet -and -not $argument.StartsWith("-")) {
        $game = $argument
        $gameWasSet = $true
        continue
    }

    $workerArgs += $argument
}

if ([string]::IsNullOrWhiteSpace($game)) {
    throw "Game id cannot be empty."
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Missing Python environment: $python"
}
if (-not (Test-Path -LiteralPath $worker -PathType Leaf)) {
    throw "Missing UB worker: $worker"
}

$keyFile = @(
    (Join-Path $workspace ".env2"),
    (Join-Path $workspace ".env2.txt")
) | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1

if (-not $keyFile) {
    throw "Missing .env2.txt or .env2. Add the existing ARC key before running UB."
}

$keyLines = @(Get-Content -LiteralPath $keyFile | ForEach-Object { $_.Trim() } | Where-Object {
    $_ -and -not $_.StartsWith("#")
})
$arcKey = $null

foreach ($line in $keyLines) {
    if ($line -match "^(?:export\s+)?ARC_API_KEY\s*=\s*(.+)$") {
        $arcKey = $Matches[1].Trim()
        break
    }
}

if (-not $arcKey -and $keyLines.Count -eq 1 -and $keyLines[0] -notmatch "=") {
    $arcKey = $keyLines[0]
}

if ($arcKey -and $arcKey.Length -ge 2) {
    $first = $arcKey.Substring(0, 1)
    $last = $arcKey.Substring($arcKey.Length - 1, 1)
    if (($first -eq '"' -and $last -eq '"') -or ($first -eq "'" -and $last -eq "'")) {
        $arcKey = $arcKey.Substring(1, $arcKey.Length - 2)
    }
}

if ([string]::IsNullOrWhiteSpace($arcKey)) {
    throw "No ARC_API_KEY was found in $([IO.Path]::GetFileName($keyFile))."
}

if (-not $noViewer) {
    if (-not (Test-Path -LiteralPath $viewer -PathType Leaf)) {
        throw "Viewer requested but missing: $viewer"
    }

    $viewerRunning = $false
    try {
        $escapedGame = [regex]::Escape($game)
        $viewerRunning = @(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
            $_.Name -in @("python.exe", "pythonw.exe") -and
            $_.CommandLine -match "ub_viewer\.py" -and
            $_.CommandLine -match "--game(?:=|\s+[`"']?)$escapedGame(?:[`"']?\s|[`"']?$)"
        }).Count -gt 0
    }
    catch {
        # A process lookup failure should not block the run; duplicate prevention is best effort.
        $viewerRunning = $false
    }

    if (-not $viewerRunning) {
        $viewerPython = if (Test-Path -LiteralPath $pythonw -PathType Leaf) { $pythonw } else { $python }
        $viewerArguments = @("`"$viewer`"", "--game", "`"$game`"")
        Start-Process -FilePath $viewerPython -ArgumentList $viewerArguments -WorkingDirectory $workspace | Out-Null
        Write-Host "Viewer started for $game."
    }
    else {
        Write-Host "Viewer already follows $game."
    }
}

$hadPriorKey = Test-Path Env:\ARC_API_KEY
$priorKey = $env:ARC_API_KEY
$hadPriorKeySource = Test-Path Env:\ARC_API_KEY_SOURCE
$priorKeySource = $env:ARC_API_KEY_SOURCE
$exitCode = 1

try {
    $env:ARC_API_KEY = $arcKey
    $env:ARC_API_KEY_SOURCE = [IO.Path]::GetFileName($keyFile)
    Write-Host "ARC profile key loaded from $([IO.Path]::GetFileName($keyFile)); starting UB for $game."
    & $python $worker --game $game @workerArgs
    $exitCode = $LASTEXITCODE
}
finally {
    if ($hadPriorKey) {
        $env:ARC_API_KEY = $priorKey
    }
    else {
        Remove-Item Env:\ARC_API_KEY -ErrorAction SilentlyContinue
    }
    if ($hadPriorKeySource) {
        $env:ARC_API_KEY_SOURCE = $priorKeySource
    }
    else {
        Remove-Item Env:\ARC_API_KEY_SOURCE -ErrorAction SilentlyContinue
    }
    $arcKey = $null
}

exit $exitCode
