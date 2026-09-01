$path = "C:\Users\ASUS\OneDrive\Documents\GitHub\reposcore\ai_review.py"
$lines = Get-Content $path
for ($i=0; $i -lt $lines.Length; $i++) {
    if ($lines[$i] -like "        tmpdir =*") {
        $lines[$i] = "        tmpdir = tempfile.mkdtemp(prefix=f\"reposcore_{repo_full_name.replace('/', '_')}_\")"
        break
    }
}
Set-Content -Path $path -Value $lines -Encoding UTF8
