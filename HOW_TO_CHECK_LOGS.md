# How to Check Logs in PresciSE

## 1. View Currently Running Command Output

If you have a command running (like the index rebuild), you can see its output in the terminal where you ran it. The logs appear in real-time.

## 2. Check Specific Log Files

PresciSE uses `loguru` for logging, which outputs to **stdout** (your terminal). To save logs to a file, redirect the output:

```powershell
# Save logs to a file while running
python -c "from core.persistence.index_manager import IndexManager; mgr = IndexManager(); retriever = mgr.load_or_build()" 2>&1 | Tee-Object -FilePath "build_log.txt"
```

## 3. View Formula Diagnostics

After indexing completes, check the diagnostics file:

```powershell
# View formula diagnostics JSON
Get-Content data/index/formula_diagnostics.json | python -m json.tool

# Or just view as-is
Get-Content data/index/formula_diagnostics.json
```

## 4. Check Formula Extraction Sources

```powershell
# Run the diagnostic script to see extraction sources
python -c "
import pickle
with open('data/index/formula_chunks.pkl', 'rb') as f:
    chunks = pickle.load(f)
sources = {}
for c in chunks:
    src = c.get('extraction_source', 'unknown')
    sources[src] = sources.get(src, 0) + 1
print('Formula extraction sources:')
for src, count in sorted(sources.items()):
    print(f'  {src}: {count}')
"
```

## 5. View Recent PowerShell Command History

```powershell
# See recent commands
Get-History | Select-Object -Last 20

# Re-run a specific command from history
Invoke-History <id>
```

## 6. Monitor Log Files in Real-Time (if redirected to file)

```powershell
# Windows equivalent of 'tail -f'
Get-Content build_log.txt -Wait -Tail 50
```

## 7. Increase Log Verbosity (if needed)

In Python, you can set the log level:

```python
from loguru import logger
import sys

# Remove default handler and add custom one with DEBUG level
logger.remove()
logger.add(sys.stderr, level="DEBUG")

# Then run your indexing
from core.persistence.index_manager import IndexManager
mgr = IndexManager()
retriever = mgr.load_or_build()
```

## 8. View Formula Diagnostics After Build

```powershell
# Check if diagnostics exist
Test-Path data/index/formula_diagnostics.json

# View with jq if installed (better formatting)
Get-Content data/index/formula_diagnostics.json | jq .

# Or use Python
python -c "
import json
with open('data/index/formula_diagnostics.json') as f:
    diag = json.load(f)
print(f\"Total formulas: {diag['total_formula_chunks']}\")
print(f\"Stage metrics: {diag.get('stage_metrics', {})}\")
"
```

## Quick Check: Is OCR Working?

```powershell
# After rebuild, check extraction sources
python -c "
import pickle
with open('data/index/formula_chunks.pkl', 'rb') as f:
    chunks = pickle.load(f)
    
ocr_count = sum(1 for c in chunks if 'ocr' in c.get('extraction_source', '').lower())
total = len(chunks)
print(f'Total formulas: {total}')
print(f'OCR formulas: {ocr_count}')
print(f'Standard formulas: {total - ocr_count}')
"
```
