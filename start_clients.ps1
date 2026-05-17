# PowerShell script to start multiple DDFed clients
# Usage: .\start_clients.ps1 -NumClients 3 -ServerAddress "127.0.0.1:8080"
param(
    [int]$NumClients = 3,
    [string]$ServerAddress = "127.0.0.1:8080",
    [switch]$IsBenign = $true,
    [switch]$SkipEncryption = $false
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Starting $NumClients DDFed Clients" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Server Address: $ServerAddress" -ForegroundColor Yellow
Write-Host "Clients: $NumClients" -ForegroundColor Yellow
Write-Host "Benign: $IsBenign" -ForegroundColor Yellow
Write-Host ""

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

$clientProcesses = @()

for ($i = 1; $i -le $NumClients; $i++) {
    Write-Host "[$i/$NumClients] Starting client-$i..." -ForegroundColor Yellow
    
    $clientArgs = @(
        "$projectRoot\client\ddfed_client_main.py",
        "--server-address", $ServerAddress,
        "--client-id", $i.ToString(),
        "--clip-norm", "1.0",
        "--threshold", "0.5"
    )
    
    if ($IsBenign) {
        $clientArgs += "--is-benign"
    }
    
    # Start client in new window
    $process = Start-Process python -ArgumentList $clientArgs -WindowStyle Normal -PassThru
    $clientProcesses += $process
    
    Write-Host "  Client-$i started (PID: $($process.Id))" -ForegroundColor Gray
    Start-Sleep -Seconds 1  # Stagger client starts
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "All $NumClients clients started!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Client PIDs:" -ForegroundColor Yellow
for ($i = 0; $i -lt $clientProcesses.Count; $i++) {
    Write-Host "  Client-$($i+1): PID $($clientProcesses[$i].Id)" -ForegroundColor Gray
}
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Wait a few seconds for clients to initialize" -ForegroundColor White
Write-Host "  2. Start the server:" -ForegroundColor White
Write-Host "     python server/ddfed_server.py --num-clients $NumClients --num-rounds 3 --skip-encryption" -ForegroundColor Gray
Write-Host ""
Write-Host "To stop all clients, press Ctrl+C or close their windows." -ForegroundColor Yellow
