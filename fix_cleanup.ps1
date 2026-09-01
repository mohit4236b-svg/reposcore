$path = "C:\Users\ASUS\OneDrive\Documents\GitHub\reposcore\ai_review.py"
$lines = Get-Content $path
# Find function start and end
$startIdx = -1
$endIdx = -1
for ($i=0; $i -lt $lines.Length; $i++) {
    if ($lines[$i] -like 'def clone_repo_bounded*') { $startIdx = $i; break }
}
if ($startIdx -eq -1) { throw "Function start not found" }
for ($i=$startIdx+1; $i -lt $lines.Length; $i++) {
    if ($lines[$i].Trim() -eq '# Prompt for AI review') { $endIdx = $i; break }
}
if ($endIdx -eq -1) { throw "Prompt line not found" }
# We will build new lines
$newLines = @()
for ($i=0; $i -lt $lines.Length; $i++) {
    if ($i -ge $startIdx -and $i -lt $endIdx) {
        # Inside function
        $line = $lines[$i]
        $trimmed = $line.Trim()
        if ($trimmed -eq 'import os') {
            $newLines += $line
            # Insert import stat with same indentation
            $indent = $line.Substring(0, $line.Length - $line.TrimStart().Length)
            $newLines += $indent + 'import stat'
        } elseif ($trimmed -eq 'import shutil') {
            $newLines += $line
            # Insert helper function with same indentation
            $indent = $line.Substring(0, $line.Length - $line.TrimStart().Length)
            $newLines += $indent + '# Helper to handle read-only files on Windows'
            $newLines += $indent + 'def _remove_readonly(func, path, exc_info):'
            $newLines += $indent + '    os.chmod(path, stat.S_IWRITE)'
            $newLines += $indent + '    func(path)'
        } elseif ($line -like '*shutil.rmtree(tmpdir, ignore_errors=True)*') {
            # Replace the line
            $indent = $line.Substring(0, $line.Length - $line.TrimStart().Length)
            $newLines += $indent + 'shutil.rmtree(tmpdir, onerror=_remove_readonly)'
        } else {
            $newLines += $line
        }
    } else {
        $newLines += $line
    }
}
Set-Content -Path $path -Value $newLines -Encoding UTF8
