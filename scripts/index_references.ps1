# --- AESTHETIC reference image indexer + renamer ---
# Renames images in by_film/by_dp to: <folder_name>_#####.<ext>
# Builds references_files.csv with scope, group, rel_path, sha1, width, height, etc.
# Re-runnable; CSV is regenerated each run.

# --- config ---
\E:\AestheticApp = "E:\AestheticApp"
\E:\AestheticApp\aesthetic\data\references = Join-Path \E:\AestheticApp "aesthetic\data\references"
\E:\AestheticApp\aesthetic\data\references\by_film = Join-Path \E:\AestheticApp\aesthetic\data\references "by_film"
\E:\AestheticApp\aesthetic\data\references\by_dp   = Join-Path \E:\AestheticApp\aesthetic\data\references "by_dp"
\ = Join-Path \E:\AestheticApp\aesthetic\data\references "references_files.csv"

# --- helpers ---
function Get-Sha1Hex {
  param([string]\)
  \ = [System.Security.Cryptography.SHA1]::Create()
  \ = [System.IO.File]::OpenRead(\)
  try {
    \ = \.ComputeHash(\)
    -join (\ | ForEach-Object { \.ToString("x2") })
  } finally { \.Close(); \.Dispose() }
}

function Get-ImageMeta {
  param([string]\)
  try {
    Add-Type -AssemblyName System.Drawing -ErrorAction SilentlyContinue | Out-Null
    \ = [System.Drawing.Image]::FromFile(\)
    \ = \.Width; \ = \.Height
    \.Dispose()
    return @{ width = \; height = \ }
  } catch {
    return @{ width = ""; height = "" }
  }
}

function Sanitize-ForFilename {
  param([string]\)
  \ = \ -replace '[^\w\-\.\(\) ]',''
  \ = \ -replace '\s+','_'
  return \
}

function Parse-YearFromFilmFolder {
  param([string]\)
  \ = [regex]::Match(\, '\((\d{4})\)$')
  if (\.Success) { return \.Groups[1].Value } else { return "" }
}

# --- init CSV ---
"scope,group,year,filename,rel_path,ext,index,width,height,sha1" | Set-Content -Encoding UTF8 \

function Process-Parent {
  param([string]\, [string]\)  # "film" | "dp"
  if (-not (Test-Path \)) { return }

  Get-ChildItem -LiteralPath \ -Directory | ForEach-Object {
    \ = \
    \ = \.Name
    \ = Sanitize-ForFilename \
    \1984 = if (\ -eq "film") { Parse-YearFromFilmFolder \ } else { "" }

    # Gather candidate images (common formats). Skip readme files.
    \ = Get-ChildItem -LiteralPath \.FullName -File |
      Where-Object {
        \.Name -notlike "README.*" -and
        (\.Extension -imatch '^\.(jpg|jpeg|png|tif|tiff|bmp)$')
      } | Sort-Object Name

    # Numbering + rename
    \100 = 0
    foreach (\ in \) {
      \100++
      \ = \.Extension.ToLower()
      \ = "{0:D5}" -f \100
      \ = "\" + "_" + "\" + "\"
      \ = Join-Path \.FullName \

      if (\.Name -ne \) {
        while (Test-Path \) {
          \100++; \ = "{0:D5}" -f \100
          \ = "\" + "_" + "\" + "\"
          \ = Join-Path \.FullName \
        }
        Rename-Item -LiteralPath \.FullName -NewName \ -Force
        \ = Get-Item -LiteralPath \
      }

      # Meta + hash
      \ = Get-ImageMeta -Path \.FullName
      \ = Get-Sha1Hex -Path \.FullName

      # CSV row
      \ = \.FullName.Replace(\E:\AestheticApp\aesthetic\data\references, "").TrimStart("\","/")
      \100,film,"Once Upon a Time in America",1984,"Tonino Delli Colli",,seed = '{0},{1},{2},{3},{4},{5},{6},{7},{8}' -f 
        \,('"' + \.Replace('"','""') + '"'),\1984, 
        \.Name,('"' + \.Replace('"','""') + '"'),\.TrimStart('.'),\100, 
        \.width,\.height, \
      Add-Content -Encoding UTF8 -Path \ -Value \100,film,"Once Upon a Time in America",1984,"Tonino Delli Colli",,seed
    }
  }
}

# --- run ---
Process-Parent -ParentPath \E:\AestheticApp\aesthetic\data\references\by_film -Scope "film"
Process-Parent -ParentPath \E:\AestheticApp\aesthetic\data\references\by_dp   -Scope "dp"
Write-Host "Indexed and renamed images. CSV -> \"
