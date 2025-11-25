# OpenAI Insights Feature

## Overview

The FastAPI application now includes an AI-powered insights endpoint that uses OpenAI's GPT models to analyze user data and generate actionable insights, patterns, and recommendations.

## Features

- **Automated Data Analysis**: Sends user data to OpenAI for intelligent analysis
- **Pattern Detection**: AI identifies trends and patterns in your user base
- **Actionable Insights**: Receive recommendations based on data analysis
- **Rate-Limited & Secure**: Protected by API key authentication and rate limiting
- **Token Tracking**: Monitor OpenAI API token usage

## Setup

### 1. Install OpenAI Package

```bash
pip install openai
```

### 2. Set Environment Variables

```bash
# Required: Your OpenAI API key
export OPENAI_API_KEY="sk-your-actual-openai-key-here"

# Optional: Choose your OpenAI model (default: gpt-3.5-turbo)
export OPENAI_MODEL="gpt-3.5-turbo"
# or for faster/cheaper analysis:
# export OPENAI_MODEL="gpt-4-turbo-preview"
# export OPENAI_MODEL="gpt-4"

# Your API key for accessing endpoints (set earlier)
export SECURE_EXPORT_API_KEY="your-secure-key"
```

### 3. Start the Server

```bash
uvicorn check:app --reload
```

## API Endpoint

### GET `/insights/users`

Generate AI insights about your user data.

**Authentication**: Required (X-API-KEY header)

**Query Parameters**:
- `limit` (int, default: 100, max: 1000): Maximum number of users to analyze
- `offset` (int, default: 0): Pagination offset
- `name_contains` (string, optional): Filter users by name substring
- `email_contains` (string, optional): Filter users by email substring

**Response** (200 OK):
```json
{
  "insights": "AI-generated analysis and insights about the user base...",
  "summary": "First 500 chars of the data summary...",
  "model": "gpt-3.5-turbo",
  "tokens_used": 487,
  "generated_at": "2025-11-25T10:30:45.123456",
  "user_count": 50
}
```

**Error Responses**:
- `401 Unauthorized`: Missing or invalid API key
- `429 Too Many Requests`: Rate limit exceeded
- `404 Not Found`: No users found matching criteria
- `503 Service Unavailable`: OpenAI client not initialized (missing API key or package)
- `500 Internal Server Error`: OpenAI API call failed

## Example Usage

### Using cURL

```bash
# Basic request
curl -H "X-API-KEY: your-secure-key" \
  "http://localhost:8000/insights/users?limit=50"

# With filters
curl -H "X-API-KEY: your-secure-key" \
  "http://localhost:8000/insights/users?limit=100&name_contains=smith&offset=0"
```

### Using Python

```python
import requests

headers = {"X-API-KEY": "your-secure-key"}
params = {
    "limit": 50,
    "name_contains": "alice"
}

response = requests.get(
    "http://localhost:8000/insights/users",
    headers=headers,
    params=params
)

if response.status_code == 200:
    insights = response.json()
    print("AI Insights:")
    print(insights["insights"])
    print(f"\nTokens used: {insights['tokens_used']}")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

### Using JavaScript/Fetch

```javascript
const apiKey = "your-secure-key";
const params = new URLSearchParams({
  limit: 50,
  name_contains: "john"
});

fetch(`http://localhost:8000/insights/users?${params}`, {
  method: "GET",
  headers: {
    "X-API-KEY": apiKey,
    "Content-Type": "application/json"
  }
})
.then(res => res.json())
.then(data => {
  console.log("AI Insights:", data.insights);
  console.log("Tokens used:", data.tokens_used);
})
.catch(err => console.error("Error:", err));
```

## How It Works

1. **Data Fetching**: Retrieves user data from SQLite DB with optional filters
2. **Report Generation**: Formats data into a human-readable report
3. **AI Analysis**: Sends report to OpenAI with analysis prompt
4. **Insight Extraction**: Parses OpenAI response for insights
5. **Response**: Returns structured insights with metadata

## Data Flow

```
User Request
    ↓
API Key Validation
    ↓
Rate Limit Check
    ↓
Database Query (user data)
    ↓
Format Data for OpenAI
    ↓
Call OpenAI API (GPT-3.5/4)
    ↓
Extract & Parse Insights
    ↓
Return JSON Response
```

## Configuration Reference

| Environment Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | (required) | Your OpenAI API key from https://platform.openai.com |
| `OPENAI_MODEL` | gpt-3.5-turbo | OpenAI model for analysis (gpt-3.5-turbo, gpt-4, gpt-4-turbo-preview) |
| `SECURE_EXPORT_API_KEY` | change-me | API key for securing endpoints |
| `SECURE_EXPORT_LOGLEVEL` | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |

## Logging

The application logs all OpenAI interactions:

```
INFO: Sending data to OpenAI for analysis...
INFO: Successfully received insights from OpenAI
INFO: Insights report generated successfully with 487 tokens used
```

Enable debug logging to see full prompts and responses:
```bash
export SECURE_EXPORT_LOGLEVEL="DEBUG"
```

## Cost Considerations

OpenAI API usage is billed per token. Monitor your usage:

- **GPT-3.5-turbo**: ~$0.0015 per 1K input tokens, ~$0.002 per 1K output tokens
- **GPT-4**: ~$0.03 per 1K input tokens, ~$0.06 per 1K output tokens
- **GPT-4-turbo**: ~$0.01 per 1K input tokens, ~$0.03 per 1K output tokens

Each request shows `tokens_used` in the response. Check your OpenAI dashboard for billing details.

## Troubleshooting

### "OPENAI_API_KEY not set in environment"
**Solution**: Set your OpenAI API key before starting the server:
```bash
export OPENAI_API_KEY="sk-..."
```

### "openai library not installed"
**Solution**: Install the package:
```bash
pip install openai
```

### "Insights feature unavailable" (503)
**Solution**: Ensure both OPENAI_API_KEY is set AND the openai package is installed.

### "Rate limit exceeded" (429)
**Solution**: Wait for the rate limit window to reset (default: 60 seconds per API key).

### "Failed to generate insights" (500)
**Solution**: Check your OpenAI API key validity and account credits. Review logs for details.

### Slow Response Times
**Solution**: The OpenAI API may take 1-5 seconds per request. Use GPT-3.5-turbo for faster responses.

## Next Steps

- Monitor OpenAI token usage and optimize prompts
- Add caching for repeated analyses
- Integrate insights into dashboards
- Export insights reports to PDF/Excel
- Set up scheduled insight generation jobs

## Security Notes

⚠️ **Important Security Reminders**:

1. **Never commit OpenAI API keys** to version control
2. **Use environment variables** for all secrets
3. **Rotate API keys** regularly
4. **Monitor API key usage** for suspicious activity
5. **Implement request logging** for audit trails
6. **Use strong API keys** for SECURE_EXPORT_API_KEY

## Support

For OpenAI API issues, visit: https://platform.openai.com/docs/guides/gpt

For application issues, check the logs:
```bash
# View last 50 lines of logs
tail -n 50 /path/to/logfile
```
