# Collects everything Windows has ALREADY recorded about a ScanReceipts crash.
#
# Run in PowerShell (admin not required, but admin reads more of the machine-wide
# WER archive), then send the file it names. Nothing is changed or deleted.

$ErrorActionPreference = 'Continue'
$out = Join-Path ([Environment]::GetFolderPath('Desktop')) 'ScanReceipts-crash-report.txt'
$since = (Get-Date).AddDays(-14)
"ScanReceipts crash report, generated $(Get-Date -Format s)" | Out-File $out -Encoding utf8

"`n=== 1. Application event log ===" | Out-File $out -Append -Encoding utf8
Get-WinEvent -FilterHashtable @{LogName='Application'; StartTime=$since} -ErrorAction SilentlyContinue |
  Where-Object { $_.Message -match 'ScanReceipts' } |
  Select-Object TimeCreated, Id, ProviderName, Message |
  Format-List | Out-File $out -Append -Encoding utf8

"`n=== 2. Reliability history ===" | Out-File $out -Append -Encoding utf8
Get-CimInstance Win32_ReliabilityRecords -ErrorAction SilentlyContinue |
  Where-Object { $_.ProductName -match 'ScanReceipts' -or $_.Message -match 'ScanReceipts' } |
  Select-Object TimeGenerated, SourceName, ProductName, Message |
  Format-List | Out-File $out -Append -Encoding utf8

# Both hives matter: the per-user tree is usually empty while the machine-wide
# one under ProgramData holds the actual reports.
"`n=== 3. Windows Error Reporting archive ===" | Out-File $out -Append -Encoding utf8
$werRoots = @(
  "$env:LOCALAPPDATA\Microsoft\Windows\WER",
  "$env:PROGRAMDATA\Microsoft\Windows\WER"
)
foreach ($root in $werRoots) {
  Get-ChildItem -Path $root -Recurse -Include '*.wer' -ErrorAction SilentlyContinue |
    Where-Object {
      (Get-Content $_.FullName -Raw -ErrorAction SilentlyContinue) -match 'ScanReceipts'
    } |
    ForEach-Object {
      "--- $($_.FullName) ---" | Out-File $out -Append -Encoding utf8
      Get-Content $_.FullName -ErrorAction SilentlyContinue |
        Select-String 'AppName|AppPath|AppVersion|ModName|ModVersion|Exception|Sig\[|EventType' |
        Out-File $out -Append -Encoding utf8
    }
}

"`n=== 4. Crash dumps ===" | Out-File $out -Append -Encoding utf8
foreach ($d in @("$env:LOCALAPPDATA\ScanReceipts\dumps", "$env:LOCALAPPDATA\CrashDumps")) {
  "--- $d ---" | Out-File $out -Append -Encoding utf8
  Get-ChildItem $d -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match 'ScanReceipts' } |
    Select-Object LastWriteTime, Length, FullName |
    Format-List | Out-File $out -Append -Encoding utf8
}

"`n=== 5. ScanReceipts logs ===" | Out-File $out -Append -Encoding utf8
Get-ChildItem "$env:LOCALAPPDATA\ScanReceipts\logs" -ErrorAction SilentlyContinue |
  Select-Object LastWriteTime, Length, Name |
  Format-List | Out-File $out -Append -Encoding utf8

Write-Host "Written to $out"
Write-Host 'Send that file, plus anything listed under sections 4 and 5.'
