# Local Model + Cloudflare Deploy (MVP)

This MVP runs modelling locally and deploys static JSON/UI to Cloudflare Pages.

## 1) Betfair cert setup (required for live odds)

Generate cert/key locally:

```bash
mkdir -p configs/certs
openssl req -x509 -nodes -newkey rsa:2048 \
  -keyout configs/certs/client.key \
  -out configs/certs/client.crt \
  -days 365
```

Upload `configs/certs/client.crt` in Betfair developer portal:
- My Security -> Non-Interactive (bot) login certificate.

Notes:
- `configs/certs/` is gitignored.
- `fetch-odds` now writes `data/raw/betfair/coverage_report.json` with status/coverage diagnostics.

## 2) Local run commands

One-off publish:

```bash
set -a; source .env; set +a
SCORING_COURSES="Catterick,Thurles" .venv/bin/python scripts/run_local_publish.py
```

Continuous updates (every 3 minutes, racecards every ~30 minutes):

```bash
set -a; source .env; set +a
SCORING_COURSES="Catterick,Thurles" .venv/bin/python scripts/run_local_publish.py --loop
```

Output JSON for the site is written to:
- `site/public/predictions.json`

## 3) Cloudflare Pages deploy from GitHub

Workflow file:
- `.github/workflows/deploy-cloudflare-pages.yml`

Set these in GitHub repo settings:
- Secret: `CLOUDFLARE_API_TOKEN`
- Secret: `CLOUDFLARE_ACCOUNT_ID`
- Variable: `CLOUDFLARE_PAGES_PROJECT`

Deploy behavior:
- Pushes to `main` that touch `site/**` trigger deploy.
- Manual deploy is available via workflow dispatch.

## 4) MVP operating model

1. Run local publish (one-off or loop).
2. Commit/push updated `site/public/predictions.json`.
3. GitHub Action deploys updated site to Cloudflare Pages.
