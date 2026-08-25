#!/usr/bin/env python3
"""Fix indentation issues in app.py - comprehensive fix"""
import py_compile

with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix render_component_bar (lines 111-122)
# Need: docstring at 4 spaces, pct at 4 spaces, color at 4 spaces
lines[111] = '    """Renders a labeled progress bar for a single component score."""\n'
lines[112] = '    pct = value / max_value\n'
lines[113] = '    color = "#27ae60" if pct >= 0.7 else "#d4a017" if pct >= 0.4 else "#c0392b"\n'

# Fix render_verdict_banner (lines 124-134)
# icon at 4 spaces, label at 4 spaces
for i in range(124, 135):
    if 'icon = ' in lines[i]:
        lines[i] = '    icon = "✅" if prediction == 1 else "⚠️"\n'
    if 'label = "High Quality Repository"' in lines[i]:
        lines[i] = '    label = "High Quality Repository" if prediction == 1 else "Low Quality / Unmaintained Repository"\n'

# Fix check_exceptions area (lines 326-328)
lines[325] = '            warning_messages = exceptions.copy()\n'
lines[326] = '            if low_confidence:\n'
lines[327] = '                warning_messages.append("Low confidence prediction (probability near 0.5).")\n'

# Fix component_bars_html area (find the for loop and pct line)
for i, line in enumerate(lines):
    if 'pct = value / 100' in line and line.strip() == 'pct = value / 100':
        if line.startswith('                                        '):  # 40 spaces
            lines[i] = '                    pct = value / 100\n'
            print(f"Fixed pct line at {i+1}")
            break

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Done. Checking syntax...")
try:
    py_compile.compile('app.py', doraise=True)
    print("Syntax OK!")
except py_compile.PyCompileError as e:
    print(f"Syntax error: {e}")
    # Show the problematic lines
    with open('app.py', 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    if hasattr(e, 'lineno') and e.lineno:
        start = max(0, e.lineno - 3)
        end = min(len(all_lines), e.lineno + 2)
        for i in range(start, end):
            print(f"{i+1}: {repr(all_lines[i])}")