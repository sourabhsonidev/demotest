# Quick Reference: OpenAI Insights API

## Setup (One-Time)

```bash
# Install package
pip install openai

# Set environment variable
export OPENAI_API_KEY="sk-your-api-key-from-openai"
export SECURE_EXPORT_API_KEY="your-app-api-key"

# Run server
uvicorn check:app --reload
```

## API Call

### cURL
```bash
curl -H "X-API-KEY: your-app-api-key" \
  "http://localhost:8000/insights/users?limit=50"
```

### Python
```python
import requests

resp = requests.get(
    "http://localhost:8000/insights/users?limit=50",
    headers={"X-API-KEY": "your-app-api-key"}
)
print(resp.json()["insights"])
```

### JavaScript
```javascript
fetch("http://localhost:8000/insights/users?limit=50", {
  headers: {"X-API-KEY": "your-app-api-key"}
})
.then(r => r.json())
.then(d => console.log(d.insights))
```

## Response Example

```json
{
  "insights": "Key findings from your user data:\n1. Strong user growth in Q4...",
  "summary": "USER DATA REPORT\n50 users analyzed...",
  "model": "gpt-3.5-turbo",
  "tokens_used": 487,
  "generated_at": "2025-11-25T10:30:45",
  "user_count": 50
}
```

## Query Parameters

| Param | Type | Default | Max | Description |
|-------|------|---------|-----|-------------|
| `limit` | int | 100 | 1000 | Users to analyze |
| `offset` | int | 0 | - | Pagination offset |
| `name_contains` | str | null | - | Filter by name |
| `email_contains` | str | null | - | Filter by email |

## Error Codes

| Code | Meaning | Fix |
|------|---------|-----|
| 401 | Missing/invalid API key | Check X-API-KEY header |
| 404 | No users found | Adjust filters |
| 429 | Rate limit (60/min) | Wait 60 seconds |
| 503 | OpenAI not configured | Set OPENAI_API_KEY env var |
| 500 | OpenAI API error | Check API key validity |

## Cost (Approx)

- GPT-3.5-turbo: **$0.0008** per request
- GPT-4: **$0.03** per request

See `tokens_used` in response for exact usage.

## Models Available

```
gpt-3.5-turbo          # Fast, cheap (default)
gpt-4-turbo-preview    # Better quality
gpt-4                  # Best quality, expensive
```

Set with: `export OPENAI_MODEL="gpt-4"`

## Full Documentation

See `OPENAI_INSIGHTS.md` for complete guide.

## Logs

```bash
# Check if OpenAI is working
grep -i openai /var/log/app.log

# Enable debug logging
export SECURE_EXPORT_LOGLEVEL="DEBUG"
```

---

**Production Checklist**:
- [ ] Use strong API keys
- [ ] Enable HTTPS
- [ ] Monitor token usage
- [ ] Set rate limits appropriately
- [ ] Rotate keys regularly
- [ ] Add request logging
- [ ] Set up alerts for errors
