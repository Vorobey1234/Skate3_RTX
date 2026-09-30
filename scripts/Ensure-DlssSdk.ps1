# RTX build prerequisites. dlss_wgpu 2.0 links v310.4.0 of NVIDIA's DLSS SDK and generates its
# bindings with libclang against the Vulkan SDK headers.
$ErrorActionPreference = 'Stop'
if (-not $env:DLSS_SDK) {
    $sdk = Join-Path (Split-Path $PSScriptRoot -Parent) '.local/DLSS'
    if (-not (Test-Path -LiteralPath (Join-Path $sdk 'include'))) {
        & git clone --depth 1 --branch v310.4.0 https://github.com/NVIDIA/DLSS.git $sdk
        if ($LASTEXITCODE -ne 0) { throw 'Could not download the NVIDIA DLSS SDK.' }
    }
    $env:DLSS_SDK = $sdk
}
# The runtime models ship separately from the build SDK. dlss_wgpu compiles
# against the 310.4.0 headers; newer runtimes keep that ABI and add models, such
# as Ray Reconstruction preset F (RR2) in 310.9.1, which the game requests.
if (-not $env:DLSS_RUNTIME) {
    $runtime = Join-Path (Split-Path $PSScriptRoot -Parent) '.local/DLSS-runtime'
    if (-not (Test-Path -LiteralPath (Join-Path $runtime 'lib/Windows_x86_64/rel/nvngx_dlssd.dll'))) {
        & git clone --depth 1 --branch v310.9.1 https://github.com/NVIDIA/DLSS.git $runtime
        if ($LASTEXITCODE -ne 0) { throw 'Could not download the NVIDIA DLSS runtime.' }
    }
    $env:DLSS_RUNTIME = $runtime
}
if (-not $env:LIBCLANG_PATH) { $env:LIBCLANG_PATH = Join-Path $env:ProgramFiles 'LLVM/bin' }
if (-not $env:VULKAN_SDK) { throw 'VULKAN_SDK is not set. Install the LunarG Vulkan SDK.' }

# NGX loads the DLSS Super Resolution and Ray Reconstruction models from beside
# the executable. Returns the copied file names.
function Copy-DlssRuntime([string]$Destination) {
    foreach ($name in @('nvngx_dlss.dll', 'nvngx_dlssd.dll')) {
        $source = Join-Path $env:DLSS_RUNTIME "lib/Windows_x86_64/rel/$name"
        if (-not (Test-Path -LiteralPath $source)) { throw "Missing DLSS runtime: $source" }
        Copy-Item -LiteralPath $source -Destination (Join-Path $Destination $name) -Force
        $name
    }
}
