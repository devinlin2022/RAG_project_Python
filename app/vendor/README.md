# vendored assets

| file | version | source | license |
|---|---|---|---|
| `sweetalert2.min.js` | 11.26.3 | https://cdn.jsdelivr.net/npm/sweetalert2@11.26.3/dist/sweetalert2.all.min.js | MIT |

Inlined into `<head>` by `app/loading.py` so the blocking dialog works without a
CDN (Colab, offline machines, Docker). To upgrade, drop in a new build and bump
the version in this table.
