# CartPole Live Web UI

Modular front-end for the infinite CartPole stream served by `libraries.stream`.

## Layout

```
public/web/
  index.html      Page structure and markup
  css/styles.css  Local theme / gauge styles
  js/app.js       SSE telemetry, controls, waveform canvas
```

## How it is served

```bash
python scripts/run_stream.py
```

- `/` and `/index.html` → `index.html`
- `/css/*` → `css/`
- `/js/*` → `js/`
- `/stream` → MJPEG video
- `/events` → SSE telemetry
- `/api/control`, `/api/models`, `/api/stats` → JSON APIs

Tailwind utilities are loaded from CDN; project-specific styles stay in `css/styles.css`.
