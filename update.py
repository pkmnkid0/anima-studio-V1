"""Command-line updater (used by update.bat / update.sh). Close Anima Studio first.

  update.py                 use the repository saved in App settings (or the last one given)
  update.py owner/name      set the repository, then update
  update.py --check         only look, change nothing
  update.py --main          follow the latest commit on main instead of releases
"""
import sys

from app.services import updater


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    flags = {a for a in argv if a.startswith("--")}
    try:
        body = {}
        if args:
            body["repo"] = args[0]
        if "--main" in flags:
            body["channel"] = "main"
        if body:
            updater.save_settings(body)
        info = updater.check()
        print(f"Installed: {info['installed']}   Latest: {info['latest']}")
        if not info["available"]:
            print("You are up to date.")
            return 0
        if info["notes"].strip():
            print("\n" + info["notes"].strip() + "\n")
        if "--check" in flags:
            print("An update is available.")
            return 0
        if updater.is_packaged():
            print("Packaged app: download the new version from " + info["url"])
            return 1
        import time
        updater.start({"zip": info["zip"], "latest": info["latest"], "sha": info["sha"]})
        last = ""
        while True:
            p = updater.progress()
            msg = f"{p['message']} {p['percent']}%" if p["state"] not in ("done", "error") else ""
            if msg and msg != last:
                print(msg)
                last = msg
            if p["state"] == "done":
                print(p["message"] + (("\n" + p["error"]) if p["error"] else ""))
                return 0
            if p["state"] == "error":
                print("Update failed: " + p["error"])
                return 1
            time.sleep(0.4)
    except updater.UpdateError as exc:
        print(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
