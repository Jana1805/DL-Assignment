# Copies everything the API needs into a cloned Hugging Face Space repository.
#
#   .\deploy\huggingface\build_space.ps1 -SpaceDir C:\path\to\your-space
#
# Then, inside the Space folder: git add -A; git commit -m "Deploy"; git push

param([Parameter(Mandatory = $true)][string]$SpaceDir)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")

if (-not (Test-Path (Join-Path $SpaceDir ".git"))) {
    throw "'$SpaceDir' is not a git repository. Clone your Space there first."
}
if (-not (Test-Path (Join-Path $root "artifacts\model.onnx"))) {
    throw "artifacts\model.onnx is missing. Run: python -m jit.train --data dataset\apachejit_total.csv"
}

foreach ($folder in "jit", "app", "artifacts") {
    $target = Join-Path $SpaceDir $folder
    if (Test-Path $target) { Remove-Item -Recurse -Force $target }
    New-Item -ItemType Directory $target | Out-Null
}
Copy-Item (Join-Path $root "jit\__init__.py"), (Join-Path $root "jit\features.py"), (Join-Path $root "jit\artifacts.py") (Join-Path $SpaceDir "jit")
Copy-Item (Join-Path $root "app\*.py") (Join-Path $SpaceDir "app")
Copy-Item (Join-Path $root "artifacts\model.onnx"), (Join-Path $root "artifacts\metadata.json") (Join-Path $SpaceDir "artifacts")
Copy-Item (Join-Path $root "Dockerfile"), (Join-Path $root "requirements.txt") $SpaceDir
Copy-Item (Join-Path $PSScriptRoot "README.md") $SpaceDir

Write-Host "Space files written to $SpaceDir"
