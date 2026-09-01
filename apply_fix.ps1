$path = "C:\Users\ASUS\OneDrive\Documents\GitHub\reposcore\ai_review.py"
$lines = Get-Content $path
$startIdx = 19
$endIdx = 65
$newLines = @()
for ($i=0; $i -lt $lines.Length; $i++) {
    if ($i -ge $startIdx -and $i -lt $endIdx) {
        $line = $lines[$i]
        $trimmed = $line.Trim()
        if ($trimmed -eq 'import os') {
            $newLines += $line
            $indent = $line.Substring(0, $line.Length - $line.TrimStart().Length)
            $newLines += $indent + 'import stat'
        } elseif ($trimmed -eq 'import shutil') {
            $newLines += $line
            $indent = $line.Substring(0, $line.Length - $line.TrimStart().Length)
            $newLines += $indent + '# Helper to handle read-only files on Windows'
            $newLines += $indent + 'def _remove_readonly(func, path, exc_info):'
            $newLines += $indent + '    os.chmod(path, stat.S_IWRITE)'
            $newLines += $indent + '    func(path)'
        } elseif ($line -like '*shutil.rmtree(tmpdir, ignore_errors=True)*') {
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
