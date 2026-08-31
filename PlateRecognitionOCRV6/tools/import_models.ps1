[CmdletBinding()]
param(
    [ValidateSet('tiny', 'small', 'medium')]
    [string]$Tier = 'medium',

    [string]$SourceRoot = 'Q:\PPOCRV5\PPOCRV6\models',

    [switch]$All,

    [switch]$Force
)

$ErrorActionPreference = 'Stop'

$moduleRoot = Split-Path -Parent $PSScriptRoot
$tiers = if ($All) { @('tiny', 'small', 'medium') } else { @($Tier) }
$expectedSha256 = @{
    tiny = '9EF676D6ED3C88256A2D92C640C44F25B0C40947E111B14B8BE8F594091563E6'
    small = '5435FD747C9E0EFE15A96D0B378D5BD157E9492ED8FD80EDF08F30D02FA24634'
    medium = '9C09ABF0957F7968C7586464B7397B84AD2387A0497A351AF40E9ACC71B673BA'
}

foreach ($currentTier in $tiers) {
    $sourceDirectory = Join-Path $SourceRoot (Join-Path $currentTier 'rec')
    $sourceOnnx = Join-Path $sourceDirectory 'inference.onnx'
    if (-not (Test-Path -LiteralPath $sourceOnnx -PathType Leaf)) {
        throw "Recognition ONNX does not exist: $sourceOnnx"
    }
    $sourceHash = (Get-FileHash -LiteralPath $sourceOnnx -Algorithm SHA256).Hash
    if ($sourceHash -ne $expectedSha256[$currentTier]) {
        throw "Unexpected SHA-256 for PP-OCRv6 $currentTier source: $sourceHash"
    }

    $destinationDirectory = Join-Path (Join-Path $moduleRoot 'models') $currentTier
    $destinationOnnx = Join-Path $destinationDirectory 'inference.onnx'
    if (-not $Force -and (Test-Path -LiteralPath $destinationOnnx)) {
        throw "Destination ONNX already exists for tier '$currentTier'. Use -Force to replace it."
    }

    New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null
    Copy-Item -LiteralPath $sourceOnnx -Destination $destinationOnnx -Force:$Force
    $destinationHash =
        (Get-FileHash -LiteralPath $destinationOnnx -Algorithm SHA256).Hash
    if ($destinationHash -ne $expectedSha256[$currentTier]) {
        throw "Copied PP-OCRv6 $currentTier ONNX failed SHA-256 validation."
    }
    Write-Output "Imported PP-OCRv6 $currentTier recognition ONNX."
}
