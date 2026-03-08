# Obsidian MCP Tools

Connects an Obsidian vault to Open WebUI via MCP, allowing AI models to read and search your notes.

## Architecture

```
Open WebUI → mcpo (port 8000) → mcp-obsidian (stdio) → Obsidian Local REST API
```

- **mcp-obsidian** — MCP server that wraps the Obsidian REST API, spawned as a subprocess inside mcpo
- **mcpo** — Translates MCP tools into an OpenAPI endpoint Open WebUI can call

## Prerequisites

- Docker Desktop
- Obsidian with the [Local REST API](https://github.com/coddingtonbear/obsidian-local-rest-api) community plugin enabled
- Open WebUI running separately

## Setup

1. Copy your API key from Obsidian → Settings → Community Plugins → Local REST API

2. Set it in `config.json` under `env.OBSIDIAN_API_KEY`

3. Build and start the stack:
   ```bash
   docker compose build mcpo
   docker compose up -d
   ```

4. In Open WebUI, add a tool server at:
   ```
   http://host.docker.internal:8000/obsidian
   ```

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Extends mcpo image with mcp-obsidian installed |
| `docker-compose.yml` | Service definition |
| `config.json` | mcpo stdio config — spawns mcp-obsidian directly |
| `obsidian.py` | Patched Obsidian client — adds env var support for host/port/protocol |

## Notes

- `obsidian.py` patches the installed mcp-obsidian to respect `OBSIDIAN_HOST`, `OBSIDIAN_PORT`, and `OBSIDIAN_PROTOCOL` env vars (the original hardcodes `127.0.0.1:27124`)
- SSL verification is disabled by default (required for Obsidian's self-signed cert)
- stdio mode avoids the SSE/streamable_http session timeout issues (previously both would drop after ~10-15 min of inactivity due to Docker NAT and Minibridge session TTL)
- The Obsidian REST API is **case-sensitive**: directory paths must match exactly (e.g. `Zettelkasten`, not `zettelkasten`)
- Do not include a trailing slash in `dirpath` — the API client appends one automatically
