---
name: anam-test
description: Use when verifying anam.ai API connectivity, listing knowledge base folders, running test searches, or debugging upload issues. Triggers on "test the API", "check anam connection", "list folders", "login to anam".
---

# Anam Test

Verify anam.ai API connectivity and test knowledge base operations.

## Prerequisites

- `ANAM_API_KEY` environment variable set
- Virtual environment activated or use `.venv/bin/python`

## Quick Test Sequence

### 1. Verify API key is set

```bash
echo "Key set: $([ -n "$ANAM_API_KEY" ] && echo 'yes' || echo 'NO')"
echo "Key prefix: ${ANAM_API_KEY:0:8}..."
```

### 2. List folders

```python
from config import Config
from anam_client import AnamClient

config = Config(anam_api_key="$ANAM_API_KEY")
client = AnamClient(config)
folders = client.list_folders()
for f in folders:
    print(f"  {f.get('name', 'unnamed')} — {f['id']}")
```

Or as a one-liner:

```bash
.venv/bin/python -c "
from config import Config; from anam_client import AnamClient
c = AnamClient(Config.from_env())
for f in c.list_folders():
    print(f\"{f.get('name','?'):30s} {f['id']}\")
"
```

### 3. Test search

```bash
.venv/bin/python -c "
import json
from config import Config; from anam_client import AnamClient
c = AnamClient(Config.from_env())
folders = c.list_folders()
if not folders:
    print('No folders found'); exit()
fid = folders[0]['id']
print(f'Searching folder: {folders[0].get(\"name\", fid)}')
results = c.search_folder(fid, 'What is a budget?', limit=3)
for r in results:
    print(json.dumps(r, indent=2, default=str)[:500])
"
```

### 4. Test upload (dry run)

```bash
.venv/bin/python cli.py upload ./some-docs/ \
  --api-key $ANAM_API_KEY \
  --llm-key $ANTHROPIC_API_KEY \
  --dry-run
```

### 5. Run eval with anam vector search

```bash
.venv/bin/python eval/run_eval.py rag-files-YYYYMMDD-HHMMSS/ \
  --llm-key $ANTHROPIC_API_KEY \
  --anam-key $ANAM_API_KEY
```

## API Endpoints

| Operation | Method | Endpoint |
|-----------|--------|----------|
| List folders | GET | `/v1/knowledge/groups` |
| Create folder | POST | `/v1/knowledge/groups` |
| Get folder | GET | `/v1/knowledge/groups/{id}` |
| Search folder | POST | `/v1/knowledge/groups/{id}/search` |
| Upload document | POST (multipart) | `/v1/knowledge/groups/{id}/documents` |
| Create tool | POST | `/v1/tools` (type: `SERVER_RAG`) |
| Get persona | GET | `/v1/personas/{id}` |

Base URL: `https://api.anam.ai` (override with `ANAM_BASE_URL`)

**Note:** Folders are flat — no nesting. The old presigned upload flow (`presigned-upload` / `confirm-upload`) has been replaced with direct multipart POST to `/documents`.

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| 401 Unauthorized | Bad or expired API key | Check `$ANAM_API_KEY` value |
| 403 Forbidden | Key lacks permission | Verify key has KB access in anam.ai dashboard |
| 404 Not Found | Wrong folder ID or endpoint | List folders first to get valid IDs |
| Connection refused | Wrong base URL | Check `$ANAM_BASE_URL` if set |
| Empty search results | Folder has no documents or docs still PROCESSING | Wait ~30s after upload, then retry |
