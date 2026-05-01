# Security deployment notes

This app accepts untrusted PDF uploads and forwards extracted paper text/features to the configured rubric model. Deploy it behind HTTPS and keep the API key only in server-side environment variables.

## Required production settings

Set these environment variables before exposing the site publicly:

```powershell
$env:DEEPSEEK_API_KEY="your key"
$env:ALLOWED_HOSTS="your-domain.com,www.your-domain.com"
$env:TRUST_PROXY_HEADERS="1"
$env:REQUIRE_HTTPS="1"
```

Use `TRUST_PROXY_HEADERS=1` only when the app is behind a trusted reverse proxy such as Nginx, Caddy, Cloudflare, Render, Fly.io, or a managed load balancer.

## Implemented controls

- Upload allowlist: `.pdf` extension, `application/pdf` MIME type when provided, and `%PDF-` magic bytes.
- Upload size cap: request body is read in 1 MB chunks and rejected above 20 MB.
- Rate limit: 5 `/api/predict` submissions per hour per IP plus User-Agent identity.
- Temporary files: uploaded PDFs are written with random server-side names and deleted in a `finally` block after parsing/scoring, including failure paths.
- Host header protection: `TrustedHostMiddleware` uses `ALLOWED_HOSTS`; default is local development only.
- Browser headers: CSP, `X-Frame-Options`, `X-Content-Type-Options`, referrer policy, permissions policy, COOP/CORP, and HSTS when HTTPS is detected.
- Error handling: user-facing PDF parse errors do not include filesystem paths or stack traces.

## Remaining operational risks

- Without user accounts, “same person” limiting is approximate. It uses IP plus User-Agent and can be bypassed by changing networks or clients.
- HTTPS encryption is provided by the reverse proxy or hosting platform, not by Uvicorn itself.
- Uploaded paper content is sent to the configured model provider after local extraction. Make that clear in any public privacy notice.
