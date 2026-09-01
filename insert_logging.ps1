$lines = Get-Content app.py
$newLines = @(
    '      # Also log to JSON Lines file',
    '      import json',
    '      jsonl_file = os.path.join(audit_dir, "predictions.jsonl")',
    '      with open(jsonl_file, "a", encoding="utf-8") as f:',
    '          json.dump(logged_features, f)',
    '          f.write([char]10)'
)
$lines = $lines[0..302] + $newLines + $lines[303..($lines.Length-1)]
Set-Content -Path app.py -Value $lines
