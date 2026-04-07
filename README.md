---
title: Job Readiness OpenEnv
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_file: server/app.py
pinned: false
---

# Job Readiness OpenEnv

## Environment variables

- `API_BASE_URL=https://api.openai.com/v1` for OpenAI client-based LLM calls
- `OPENAI_API_KEY=...`
- `MODEL_NAME=gpt-4o-mini`
- `ENV_SERVER_URL=http://127.0.0.1:8000` for the local OpenEnv server used by `inference.py`

## Notes

The environment logic is unchanged. The update separates OpenAI API configuration from the local OpenEnv server URL so the project can use the OpenAI client without breaking `/reset` and `/step`.