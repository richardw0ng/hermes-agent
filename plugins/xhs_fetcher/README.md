# XHS Fetcher

Hermes plugin for fetching Xiaohongshu / RedNote content through the
open-source `xhs-kit` command line tool.

## Enable

Add the plugin to `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - xhs_fetcher
```

Install the optional fetcher dependency in the Hermes virtual environment:

```bash
cd /Users/richard_w0ng/code_new/hermes-agent
source .venv/bin/activate
python -m pip install xhs-kit
```

If your network is slow, use a mirror:

```bash
python -m pip install xhs-kit -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
```

The plugin prefers the `xhs-kit` Python SDK and automatically uses a local
Chrome executable when Playwright's bundled Chromium is not installed. Override
the browser path when needed:

```bash
XHS_BROWSER_BIN="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
```

## Cookies

Xiaohongshu usually requires a logged-in web cookie for stable detail and
comment access. Keep cookies out of git. Prefer one of these env values in
`~/.hermes/.env`:

```bash
XHS_COOKIES_PATH=/absolute/path/to/cookies.json
```

or use `xhs-kit`'s own login/setup flow if your installed version provides one.

## Tools

- `xhs_check_status`: checks whether `xhs-kit` is installed and whether cookie
  paths are configured.
- `xhs_search_notes`: runs `xhs-kit search -k <keyword>`.
- `xhs_fetch_note`: runs `xhs-kit detail --feed-id <id> --xsec-token <token>`.
- `xhs_fetch_comments`: fetches detail with comments and extracts comment
  arrays from the returned JSON.

The plugin intentionally wraps `xhs-kit` instead of vendoring XHS signing code.
When Xiaohongshu changes its web API, update `xhs-kit` independently.
