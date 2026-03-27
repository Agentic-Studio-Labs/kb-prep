# Lesson 2 Pipeline Run Index

This folder contains one before/after test run:

- `01-before-analyze-cli.txt` - baseline analyze + benchmark CLI output
- `01-before-manifest.json` - baseline manifest
- `01-before-web-report.html` - baseline web dashboard

- `02-fix-cli.txt` - fix step CLI output
- `fixed-output/` - fixed markdown files + sidecars from fix step

- `03-after-analyze-including-fix-report-cli.txt` - post-fix analyze CLI output (includes generated `ingestgate-fix-*.md` as input)
- `03-after-manifest-including-fix-report.json` - post-fix manifest with fix report included
- `03-after-web-report-including-fix-report.html` - post-fix dashboard with fix report included

- `04-after-clean-analyze-cli.txt` - post-fix analyze CLI output excluding `ingestgate-fix-*` report
- `04-after-clean-manifest.json` - clean post-fix manifest (apples-to-apples with baseline)
- `04-after-clean-web-report.html` - clean post-fix dashboard (recommended comparison view)
