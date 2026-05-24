param(
    [string]$Model = "qwen2.5-coder:latest",
    [string]$VenvActivateScript = $null,
    [string]$MainScript = $null
)

$startTime = Get-Date
$ScriptDir = $PSScriptRoot
$RootDir = (Resolve-Path (Join-Path $ScriptDir "..")).Path

if (-not $VenvActivateScript) {
    $VenvActivateScript = Join-Path $RootDir ".venv\Scripts\Activate.ps1"
}

if (-not $MainScript) {
    $MainScript = Join-Path $RootDir "scripts\detect_qwen.py"
}

Start-Transcript -Path (Join-Path $RootDir "detect_qwen_with_ollama.log") -Append

$ErrorActionPreference = "Stop"
$ollamaProcess = $null
$scriptSuccess = $true

function Stop-OllamaSession {
    param(
        [Parameter(Mandatory = $true)]
        [System.Diagnostics.Process]$Process,
        [Parameter(Mandatory = $true)]
        [string]$Model
    )

    if (-not $Process -or $Process.HasExited) {
        return
    }

    try {
        $Process.StandardInput.WriteLine("/bye")
        $Process.StandardInput.Flush()
        $Process.StandardInput.Close()
    }
    catch {}

    if ($Process.WaitForExit(15000)) {
        return
    }

    try {
        & ollama stop $Model | Out-Null
    }
    catch {}

    if ($Process.WaitForExit(10000)) {
        return
    }

    Write-Warning "Ollama no cerró tras /bye y stop; finalizando proceso."
    $Process.Kill()
}

try {
    Write-Host "[1/4] Iniciando Ollama en segundo plano con modelo '$Model'..."

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "ollama"
    $psi.Arguments = "run $Model"
    $psi.UseShellExecute = $false
    $psi.RedirectStandardInput = $true
    $psi.RedirectStandardOutput = $false
    $psi.RedirectStandardError = $false
    $psi.CreateNoWindow = $true

    $ollamaProcess = New-Object System.Diagnostics.Process
    $ollamaProcess.StartInfo = $psi
    [void]$ollamaProcess.Start()

    Start-Sleep -Seconds 15

    Write-Host "[2/4] Activando entorno virtual..."
    . $VenvActivateScript

    Write-Host "[3/4] Ejecutando detección de alucinaciones con Qwen..."
    python $MainScript --model $Model --overwrite

    Write-Host "[4/4] Cerrando sesión de Ollama..."
    if ($ollamaProcess -and -not $ollamaProcess.HasExited) {
        Stop-OllamaSession -Process $ollamaProcess -Model $Model
    }

    Write-Host "Proceso completado."
}
catch {
    Write-Error "Error durante la ejecución: $($_.Exception.Message)"
    $scriptSuccess = $false

    if ($ollamaProcess -and -not $ollamaProcess.HasExited) {
        try {
            Stop-OllamaSession -Process $ollamaProcess -Model $Model
        }
        catch {
            try { $ollamaProcess.Kill() } catch {}
        }
    }
}
finally {
    Stop-Transcript

    if ($scriptSuccess) {
        $endTime = Get-Date
        $elapsed = $endTime - $startTime
        $elapsedFormat = "{0:00}h {1:00}m {2:00}s" -f $elapsed.Hours, $elapsed.Minutes, $elapsed.Seconds
    } else {
        exit 1
    }
}
