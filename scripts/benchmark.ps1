# Fase 8 (D8): fija el reloj de la GPU, mide la eficiencia en WSL y SIEMPRE restablece el reloj al terminar.
# Ejecutar en PowerShell COMO ADMINISTRADOR, con el equipo sin otra carga (VS Code y demás aplicaciones cerrados):
#   powershell -ExecutionPolicy Bypass -File scripts\benchmark.ps1                # India
#   powershell -ExecutionPolicy Bypass -File scripts\benchmark.ps1 --dataset japan
#   powershell -ExecutionPolicy Bypass -File scripts\benchmark.ps1 --reanudar     # continúa tras un aborto
# El reloj fijado debe coincidir con reloj_fijo_sm_mhz de configs/bench.yaml (el script de Python lo verifica).
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Argumentos)

$ErrorActionPreference = "Stop"
$RelojMHz = 2505
$Proyecto = "/mnt/d/1-Personal/5-Studies/MaestriaIA/S2/Vision de computadora/proyecto_yolo/Codigo"

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { throw "Se requiere PowerShell como administrador para fijar el reloj de la GPU (nvidia-smi -lgc)." }

Write-Host ">> Fijando el reloj SM de la GPU en $RelojMHz MHz"
nvidia-smi -lgc "$RelojMHz,$RelojMHz"
if ($LASTEXITCODE -ne 0) { throw "nvidia-smi -lgc falló (código $LASTEXITCODE): el driver no permite fijar el reloj." }
try {
    # Docker Desktop puede quedar como distribución WSL por defecto: se usa Ubuntu explícitamente
    wsl -d Ubuntu bash "$Proyecto/scripts/benchmark.sh" @Argumentos
    $codigo = $LASTEXITCODE
}
finally {
    Write-Host ">> Restableciendo el reloj de la GPU (nvidia-smi -rgc)"
    nvidia-smi -rgc
}
exit $codigo
