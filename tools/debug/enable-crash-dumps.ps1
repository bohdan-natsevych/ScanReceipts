# Ask Windows to save a crash dump when ScanReceipts.exe dies.
#
# Run ONCE, in an ADMIN PowerShell, on the machine that reproduces the crash.
# Windows Error Reporting only honours this under HKLM, which is why it needs
# admin. The key is scoped to ScanReceipts.exe, so no other program is affected.
#
# Undo with:  .\enable-crash-dumps.ps1 -Remove

param([switch]$Remove)

$ErrorActionPreference = 'Stop'
$root = 'HKLM:\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps'
$key = Join-Path $root 'ScanReceipts.exe'
$folder = Join-Path $env:LOCALAPPDATA 'ScanReceipts\dumps'

$admin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) {
    Write-Error 'Run this in an Administrator PowerShell.'
}

if ($Remove) {
    if (Test-Path $key) { Remove-Item $key -Recurse -Force }
    Write-Host 'Crash dumps for ScanReceipts.exe are switched off again.'
    return
}

if (-not (Test-Path $key)) { New-Item -Path $key -Force | Out-Null }
if (-not (Test-Path $folder)) { New-Item -ItemType Directory -Path $folder -Force | Out-Null }

# DumpType 1 is a minidump: a few MB, and it still carries the exception code,
# the loaded modules and every thread stack. Type 2 is a full dump, which for
# this app runs to roughly a gigabyte per crash.
New-ItemProperty -Path $key -Name DumpFolder -Value $folder -PropertyType ExpandString -Force | Out-Null
New-ItemProperty -Path $key -Name DumpType -Value 1 -PropertyType DWord -Force | Out-Null
New-ItemProperty -Path $key -Name DumpCount -Value 5 -PropertyType DWord -Force | Out-Null

Write-Host "Crash dumps are on. They will appear in:"
Write-Host "  $folder"
Write-Host ''
Write-Host 'Now reproduce the crash, then send everything from:'
Write-Host "  $folder"
Write-Host "  $env:LOCALAPPDATA\ScanReceipts\logs"
