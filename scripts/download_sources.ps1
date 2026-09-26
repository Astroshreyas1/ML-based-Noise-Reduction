<#
Download the public source piles into $env:ANC_DATA_ROOT\sources (default .\data\sources).
Idempotent. Run in the background at hour 0.

  pwsh scripts/download_sources.ps1          # LibriSpeech dev-clean + train-clean-100, ESC-50
  pwsh scripts/download_sources.ps1 -All     # + Zenodo gunshot range set, RIRS_NOISES (MAD: see below)

Manual (forms / logins):
  UrbanSound8K  https://urbansounddataset.weebly.com/urbansound8k.html -> class 6 (gun_shot) wavs
                into sources\impulsive\esc50_gunshot\
  Lombard GRID  https://spandh.dcs.shef.ac.uk/avlombard/ -> audio\*.wav into sources\speech\lombardgrid\
  MAD           kaggle datasets download -d junewookim/mad-dataset-military-audio-dataset --unzip
                then: ancdata import-mad <extracted dir>
#>
param([switch]$All)
$ErrorActionPreference = "Stop"
$root = Join-Path ($env:ANC_DATA_ROOT ?? "data") "sources"
$tmp = Join-Path $env:TEMP "ancdl"
New-Item -ItemType Directory -Force $root, $tmp | Out-Null

$tar = Join-Path $env:SystemRoot 'System32' 'tar.exe'   # Windows bsdtar; Git's /usr/bin/tar misreads C: as a host
function Fetch($url, $dest) {
    # resumable (-C -), retries on any error; a partial file is continued, not restarted
    Write-Host "get  $url"
    curl.exe -L --retry 10 --retry-all-errors --retry-delay 5 -C - -o $dest $url
    if ($LASTEXITCODE -ne 0) { throw "download failed ($LASTEXITCODE): $url" }
}

# --- LibriSpeech (CC BY 4.0) ---
$ls = Join-Path $root "speech\librispeech"
New-Item -ItemType Directory -Force $ls | Out-Null
foreach ($part in @("dev-clean", "train-clean-100")) {
    if (Test-Path (Join-Path $ls "LibriSpeech\$part")) { Write-Host "have LibriSpeech/$part"; continue }
    $tgz = Join-Path $tmp "$part.tar.gz"
    Fetch "https://www.openslr.org/resources/12/$part.tar.gz" $tgz
    & $tar -xzf $tgz -C $ls
    if ($LASTEXITCODE -ne 0) { throw "extract failed: $tgz" }
}

# --- ESC-50 (CC BY-NC 3.0) ---
$esc = Join-Path $root "noise\esc50"
if (-not (Test-Path (Join-Path $esc "ESC-50-master"))) {
    New-Item -ItemType Directory -Force $esc | Out-Null
    $zip = Join-Path $tmp "esc50.zip"
    Fetch "https://github.com/karolpiczak/ESC-50/archive/master.zip" $zip
    Expand-Archive -Force $zip $esc
} else { Write-Host "have ESC-50" }

if ($All) {
    # --- Multi-firearm multi-orientation gunshot set (Zenodo 7004819) ---
    $field = Join-Path $root "impulsive\field"
    New-Item -ItemType Directory -Force $field | Out-Null
    if (-not (Get-ChildItem $field -Recurse -File | Select-Object -First 1)) {
        $rec = Invoke-RestMethod "https://zenodo.org/api/records/7004819"
        foreach ($f in $rec.files) {
            $dest = Join-Path $tmp $f.key
            Fetch $f.links.self $dest
            if ($dest -like "*.zip") { Expand-Archive -Force $dest $field }
        }
    } else { Write-Host "have Zenodo gunshots" }

    # --- RIRS_NOISES (Apache 2.0), optional real RIRs ---
    $rz = Join-Path $tmp "rirs_noises.zip"
    Fetch "https://www.openslr.org/resources/28/rirs_noises.zip" $rz
    $rr = Join-Path $root "rir\real"
    New-Item -ItemType Directory -Force $rr | Out-Null
    Expand-Archive -Force $rz $rr
}
Write-Host "done. next: ancdata rir; ancdata registry; ancdata selftest --config configs/train.yaml"
