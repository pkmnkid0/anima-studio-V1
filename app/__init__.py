import os
from flask import Flask, jsonify, render_template, request

from . import config, security
from .routes import projects, collect, edit, caption, train, monitor, explore, files, analyze, pages, appsettings, update


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # never serve a stale explore.js/css after an update
    security.install(app, config.HOST)   # same-origin only: no CORS, host allow-list, safe headers

    app.register_blueprint(update.bp, url_prefix="/api/update")
    app.register_blueprint(projects.bp, url_prefix="/api/projects")
    app.register_blueprint(collect.bp, url_prefix="/api/collect")
    app.register_blueprint(explore.bp, url_prefix="/api/explore")
    app.register_blueprint(edit.bp, url_prefix="/api/edit")
    app.register_blueprint(caption.bp, url_prefix="/api/caption")
    app.register_blueprint(train.bp, url_prefix="/api/train")
    app.register_blueprint(monitor.bp, url_prefix="/api/monitor")
    app.register_blueprint(files.bp, url_prefix="/api/files")        # folder browser, thumbnails, subsets
    app.register_blueprint(analyze.bp, url_prefix="/api/analyze")    # dataset health, balance, cleanup, coverage

    _boot_backend()

    app.register_blueprint(appsettings.bp, url_prefix="/api/appsettings")

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/health")
    def health():
        return jsonify({"ok": True})

    return app


_booted = False


def _boot_backend() -> None:
    """Every launch of the app brings the training backend up IF it is already
    installed. Nothing is downloaded or installed automatically - missing
    dependencies are installed only when the user presses Install in the
    Backend/Train tab."""
    global _booted
    if _booted:
        return
    _booted = True
    import threading

    def run():
        try:
            from .services import backend_manager, backend_setup
            if backend_setup.is_installed():
                backend_manager.maybe_autostart()
        except Exception:  # noqa: BLE001 - startup must never fail because of this
            pass

    threading.Thread(target=run, daemon=True, name="backend-boot").start()
