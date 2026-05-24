Start-Transcript -Path (Join-Path $PSScriptRoot "..\full_workflow_terminal.txt") -Append

$totalStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "Ejecutando workflow completo para crear trazas con Ollama..." -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

Write-Host "`n----------------------------------------------"
Write-Host "----------------------------------------------"
Write-Host "PARTE 1: Creando trazas de ChatDev..."
Write-Host "----------------------------------------------"
Write-Host "----------------------------------------------"

$p1Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
& (Join-Path $RootDir "scripts\create_chatdev_traces_with_ollama.ps1") -Model "qwen2.5-coder:latest"

$p1Stopwatch.Stop()
$tiempoP1 = [Math]::Round($p1Stopwatch.Elapsed.TotalSeconds)

Write-Host "`n----------------------------------------------"
Write-Host "----------------------------------------------"
Write-Host "Tiempo Parte 1 (ChatDev):`t$tiempoP1 segundos"
Write-Host "----------------------------------------------"
Write-Host "----------------------------------------------"

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

Write-Host "`n----------------------------------------------"
Write-Host "----------------------------------------------"
Write-Host "PARTE 2: Ejecutando script de detección de alucinaciones..."
Write-Host "----------------------------------------------"
Write-Host "----------------------------------------------"

$p2Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
& (Join-Path $RootDir "scripts\detect_qwen_with_ollama.ps1")

$p2Stopwatch.Stop()
$tiempoP2 = [Math]::Round($p2Stopwatch.Elapsed.TotalSeconds)

$totalStopwatch.Stop()
$tiempoTotal = [Math]::Round($totalStopwatch.Elapsed.TotalSeconds)

Write-Host "`n==============================================" -ForegroundColor Green
Write-Host "Workflow completo ejecutado con éxito." -ForegroundColor Green
Write-Host "----------------------------------------------"
Write-Host "RESUMEN DE TIEMPOS"
Write-Host "Parte 1 (ChatDev):`t$tiempoP1 segundos"
Write-Host "Parte 2 (Detección):`t$tiempoP2 segundos"
Write-Host "Tiempo total:`t`t$tiempoTotal segundos"
Write-Host "----------------------------------------------"
Write-Host "==============================================" -ForegroundColor Green
Stop-Transcript
