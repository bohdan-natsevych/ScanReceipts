; Per-user installer for Scan Receipts. Built by .github/workflows/release.yml.
; AppVersion is supplied on the command line: iscc /DAppVersion=0.1.2 ScanReceipts.iss

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{8F2C41D6-6B4E-4C2F-9E51-2A7D3B5C8E14}
AppName=Scan Receipts
AppVersion={#AppVersion}
AppPublisher=Scan Receipts
DefaultDirName={localappdata}\Programs\ScanReceipts
DefaultGroupName=Scan Receipts
DisableDirPage=yes
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=Output
OutputBaseFilename=ScanReceipts-Setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
RestartApplications=no
UninstallDisplayName=Scan Receipts

[Files]
Source: "..\dist\ScanReceipts\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\Scan Receipts"; Filename: "{app}\ScanReceipts.exe"
Name: "{userdesktop}\Scan Receipts"; Filename: "{app}\ScanReceipts.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Run]
; This entry must stay unskipped in silent mode: the in-app updater installs
; silently and expects the application to come back by itself.
Filename: "{app}\ScanReceipts.exe"; Description: "Start Scan Receipts"; Flags: nowait postinstall runasoriginaluser
