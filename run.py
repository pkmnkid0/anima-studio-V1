#!/usr/bin/env python3
"""
Anima Studio - entry point.

Run with:  python run.py
Then open the URL it prints (defaults to http://127.0.0.1:7864).

Environment variables:
  ANIMA_STUDIO_HOST   - bind host (default 127.0.0.1)
  ANIMA_STUDIO_PORT   - bind port (default 7864)
  ANIMA_STUDIO_DATA   - where datasets/presets/model cache are stored
                        (default: ./data next to this file)
"""
import webbrowser
import threading

from app import create_app
from app.services import procs  # noqa: F401  (registers the exit cleanup)
from app import config


def main():
    app = create_app()
    url = f"http://{config.HOST}:{config.PORT}"
    print(f"\n  Anima Studio is running at {url}\n  Press Ctrl+C to stop.\n")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=config.HOST, port=config.PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
