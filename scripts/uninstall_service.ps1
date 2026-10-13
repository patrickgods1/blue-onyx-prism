#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Removes the Blue Onyx Prism Windows service, its event log source and firewall rule, and
    restores the ServicesPipeTimeout that install_service.ps1 recorded before raising it.
#>
param([int]$Port = 32168)

$ErrorActionPreference = "Continue"
$ServiceName = "BlueOnyxPrismService"
$EventSource = "BlueOnyxPrism"

$svc = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if ($svc) {
    if ($svc.Status -ne "Stopped") {
        Write-Host "Stopping $ServiceName ..."
        Stop-Service $ServiceName -Force
        Start-Sleep -Seconds 2
    }
    Write-Host "Deleting $ServiceName ..."
    $out = & sc.exe delete $ServiceName
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "sc.exe delete failed (exit $LASTEXITCODE): $out"
    } else {
        for ($i = 0; $i -lt 60 -and (Get-Service -Name $ServiceName -ErrorAction SilentlyContinue); $i++) {
            Start-Sleep -Milliseconds 500
        }
        if (Get-Service -Name $ServiceName -ErrorAction SilentlyContinue) {
            Write-Warning "$ServiceName is marked for deletion but still registered: another program (services.msc, Event Viewer) holds it open. It disappears when that program is closed or after a reboot."
        }
    }
} else {
    Write-Host "Service $ServiceName not installed."
}

$ruleName = "Blue Onyx Prism ($Port)"
if (Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue) {
    Remove-NetFirewallRule -DisplayName $ruleName
    Write-Host "Removed firewall rule '$ruleName'."
}

# ServicesPipeTimeout: put back the value recorded at install (-1 = it was not set), unless it was
# changed since (then someone else relies on the new value).
$ControlKey = "HKLM:\SYSTEM\CurrentControlSet\Control"
$StateKey = "HKLM:\SOFTWARE\BlueOnyxPrism"
$PipeTimeoutMs = 600000
$recorded = (Get-ItemProperty -Path $StateKey -Name PreviousServicesPipeTimeout -ErrorAction SilentlyContinue).PreviousServicesPipeTimeout
if ($null -ne $recorded) {
    $current = (Get-ItemProperty -Path $ControlKey -Name ServicesPipeTimeout -ErrorAction SilentlyContinue).ServicesPipeTimeout
    if ($null -ne $current -and [int64]$current -eq $PipeTimeoutMs -and [int64]$recorded -ne $PipeTimeoutMs) {
        if ([int64]$recorded -lt 0) {
            Remove-ItemProperty -Path $ControlKey -Name ServicesPipeTimeout
            Write-Host "Removed ServicesPipeTimeout (it was not set before the install; applies after a reboot)."
        } else {
            Set-ItemProperty -Path $ControlKey -Name ServicesPipeTimeout -Value ([int]$recorded) -Type DWord
            Write-Host "Restored ServicesPipeTimeout to $recorded ms (applies after a reboot)."
        }
    } elseif ($null -ne $current -and [int64]$current -ne $PipeTimeoutMs) {
        Write-Host "ServicesPipeTimeout was changed to $current ms after the install; left as is."
    }
    Remove-Item -Path $StateKey -Recurse -Force
} else {
    Write-Host "No recorded ServicesPipeTimeout (installed by an older script); left as is. Remove HKLM\SYSTEM\CurrentControlSet\Control\ServicesPipeTimeout by hand if nothing else needs it."
}

try {
    if ([System.Diagnostics.EventLog]::SourceExists($EventSource)) {
        Remove-EventLog -Source $EventSource
        Write-Host "Removed event log source $EventSource."
    }
} catch { Write-Warning "Could not remove event log source: $_" }

Write-Host "Done. The install dir keeps its ACL (Administrators/SYSTEM full, Users read) and data;"
Write-Host "delete the folder to remove everything."
