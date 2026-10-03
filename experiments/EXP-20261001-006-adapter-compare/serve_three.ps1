$ErrorActionPreference = "Stop"
$py = "C:\Users\okevi\SJSU\DATA298\298P\host_finetune\.venv\Scripts\python.exe"
$root = "C:\Users\okevi\SJSU\DATA298\298P"
Set-Location $root
$env:SKIP_OLLAMA_SMOKE_TEST = "1"
$jobs = @(
    @{ Name = "lemkin-cleaned"; Adapter = "$root\host_finetune\output\lemkin_lora_cleaned"; Gguf = "$root\host_finetune\output\lemkin-cleaned" },
    @{ Name = "lemkin-balanced"; Adapter = "$root\host_finetune\output\lemkin_lora_balanced"; Gguf = "$root\host_finetune\output\lemkin-balanced" },
    @{ Name = "lemkin-r32"; Adapter = "$root\host_finetune\output\lemkin_lora_r32"; Gguf = "$root\host_finetune\output\lemkin-r32" }
)
foreach ($job in $jobs) {
    Write-Output ("SERVE " + $job.Name)
    $env:LORA_ADAPTER_PATH = $job.Adapter
    $env:GGUF_DIR = $job.Gguf
    $env:OLLAMA_MODEL_NAME = $job.Name
    & $py -m host_finetune.merge_and_export
    if ($LASTEXITCODE -ne 0) { throw "merge failed for $($job.Name)" }
    & $py -m host_finetune.register_ollama
    if ($LASTEXITCODE -ne 0) { throw "register failed for $($job.Name)" }
}
Write-Output "SERVE complete"
