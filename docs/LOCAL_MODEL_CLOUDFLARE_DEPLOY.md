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
SCORING_COURSES="Cheltenham" .venv/bin/python scripts/run_local_publish.py
```

Continuous updates (every 3 minutes, racecards every ~30 minutes):

```bash
set -a; source .env; set +a
SCORING_COURSES="Cheltenham" .venv/bin/python scripts/run_local_publish.py --loop
```

Output JSON for the site is written to:
- `site/public/predictions.json`

## 3) Cloudflare Pages deploy from GitHub

Workflow file:
- `.github/workflows/deploy-cloudflare-pages.yml`

### 3.1 Create the Pages project (first time only)
1. In Cloudflare dashboard, go to **Workers & Pages** -> **Create application** -> **Pages**.
2. Create project name (for example `cheltenham-predictions`).
3. You can skip Cloudflare's Git integration because deploys will come from GitHub Actions.
4. Copy the project name exactly.

### 3.2 Create API token + account ID
1. Cloudflare -> **My Profile** -> **API Tokens** -> **Create Token** -> **Custom token**.
2. Minimum permissions:
- `Account` -> `Cloudflare Pages` -> `Edit`
- `Account` -> `Account Settings` -> `Read`
3. Account resources: include your target account.
4. Save the token once (you won't be able to view it again).
5. Get Account ID from Cloudflare dashboard sidebar (**Account ID**).

### 3.3 Add GitHub secrets/variables
In GitHub repo -> **Settings** -> **Secrets and variables** -> **Actions**:
- Secret: `CLOUDFLARE_API_TOKEN` (the token above)
- Secret: `CLOUDFLARE_ACCOUNT_ID` (Cloudflare account ID)
- Variable: `CLOUDFLARE_PAGES_PROJECT` (Pages project name)
- Optional variable: `CLOUDFLARE_PAGES_DEPLOY_DIR`

`CLOUDFLARE_PAGES_DEPLOY_DIR` options:
- Omit it: deploy full built site (`site/dist`) (default)
- Set to `site/public`: deploy static JSON-only payloads (including `predictions.json`)

### 3.4 Deploy behavior
- Pushes to `main` that touch `site/**` or the workflow file trigger deploy.
- Manual deploy is available via **Actions** -> **Deploy Cloudflare Pages** -> **Run workflow**.

## 4) MVP operating model

1. Run local publish (one-off or loop).
2. Commit/push updated `site/public/predictions.json`.
3. GitHub Action deploys to Cloudflare Pages.
4. Verify deployment URL in the workflow logs.
