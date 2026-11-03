param(
    [string]$Source = 'F:\LatentPort\runtime\native\llama.cpp',
    [string]$Build = 'F:\LatentPort\runtime\native\build',
    [string]$Python = 'python',
    [string]$CudaArch = '120',
    [int]$Jobs = 2,
    [string]$VsRoot = 'C:\Program Files\Microsoft Visual Studio\18\Insiders'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$sourcePath = [IO.Path]::GetFullPath($Source)
$buildPath = [IO.Path]::GetFullPath($Build)
if (-not $sourcePath.StartsWith('F:\LatentPort\', [StringComparison]::OrdinalIgnoreCase) -or
    -not $buildPath.StartsWith('F:\LatentPort\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Keep the isolated llama.cpp source and bulk build files under F:\LatentPort.'
}
& $Python -B "$repo\native\patch_llama.py" $sourcePath
if ($LASTEXITCODE) { throw 'Pinned source patch failed' }
$cmake = "$VsRoot\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ninja = "$VsRoot\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
$vcvars = "$VsRoot\VC\Auxiliary\Build\vcvars64.bat"
# Import compiler environment; this command performs no filesystem mutation.
$compilerEnv = & cmd.exe /d /c "call `"$vcvars`" -vcvars_ver=14.44 >nul && set"
if ($LASTEXITCODE) { throw 'MSVC environment initialization failed' }
foreach ($line in $compilerEnv) {
    if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
}
$ErrorActionPreference = 'Continue' # CMake emits normal status lines on stderr; check its exit code.
$originalTemp, $originalTmp = $env:TEMP, $env:TMP
$nativeTemp = Join-Path $buildPath 'tmp'
New-Item -ItemType Directory -Force -Path $nativeTemp -ErrorAction Stop | Out-Null
$env:TEMP = $nativeTemp
$env:TMP = $nativeTemp
try {
& $cmake -S "$repo\native" -B $buildPath -G Ninja "-DCMAKE_MAKE_PROGRAM=$ninja" `
    '-DCMAKE_BUILD_TYPE=Release' "-DLATENTPORT_LLAMA_SOURCE=$sourcePath" `
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch" '-DCMAKE_CUDA_COMPILER=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/nvcc.exe'
if ($LASTEXITCODE) { throw 'Native configuration failed' }
& $cmake --build $buildPath --parallel $Jobs --target llama latentport-layout-test
if ($LASTEXITCODE) { throw 'Native build failed' }
& "$buildPath\latentport-layout-test.exe"
if ($LASTEXITCODE) { throw 'CPU layout checks failed' }
} finally {
    $env:TEMP, $env:TMP = $originalTemp, $originalTmp
}
$bin = Join-Path $buildPath 'bin'
Write-Output "Native build complete. Set LATENTPORT_NATIVE_LIB to the llama.dll in $bin. No model or GPU execution was performed."
