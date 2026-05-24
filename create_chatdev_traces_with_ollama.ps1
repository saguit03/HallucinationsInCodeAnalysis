param(
    [string]$Model = "qwen2.5-coder",
    [string]$VenvActivateScript = $null,
    [string]$MainScript = $null
)

& (Join-Path $PSScriptRoot "scripts\create_chatdev_traces_with_ollama.ps1") @PSBoundParameters
exit $LASTEXITCODE
