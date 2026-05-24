param(
    [string]$Model = "qwen2.5-coder:latest",
    [string]$VenvActivateScript = $null,
    [string]$MainScript = $null
)

& (Join-Path $PSScriptRoot "scripts\detect_qwen_with_ollama.ps1") @PSBoundParameters
exit $LASTEXITCODE
