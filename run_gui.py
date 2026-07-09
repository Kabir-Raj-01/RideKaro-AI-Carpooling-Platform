"""
run_gui.py
══════════════════════════════════════════════════════════════
One-click launcher for the Carpooling AI Web GUI.

Usage:
    python run_gui.py
    python run_gui.py --port 8080
    python run_gui.py --no-browser      # don't auto-open browser

The script:
  1. Checks all dependencies are installed
  2. Starts the Flask server
  3. Opens the browser automatically
══════════════════════════════════════════════════════════════
"""
import sys
import os
import time
import argparse
import threading
import subprocess

def check_deps():
    missing = []
    for pkg in ["flask", "networkx", "colorama", "tabulate", "matplotlib", "numpy"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\n  [!] Missing packages: {', '.join(missing)}")
        print(f"  Installing via pip...\n")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)

def open_browser(port, delay=1.5):
    time.sleep(delay)
    url = f"http://127.0.0.1:{port}"
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser(description="Carpooling AI - Web GUI Launcher")
    parser.add_argument("--port",       type=int, default=5000, help="Port (default: 5000)")
    parser.add_argument("--no-browser", action="store_true",    help="Don't auto-open browser")
    args = parser.parse_args()

    check_deps()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    port = args.port
    url  = f"http://127.0.0.1:{port}"

    print("\n" + "═"*58)
    print("  🚗  Intelligent Carpooling Optimisation System")
    print("  NUCES · AI Lab Final Project")
    print("  Kabir Raj (23K-0702) · Hassnain Aziz (23K-0905)")
    print("═"*58)
    print(f"\n  🌐  Web GUI → {url}")
    print("  ⌨️   Press Ctrl+C to stop the server\n")

    if not args.no_browser:
        threading.Thread(target=open_browser, args=(port,), daemon=True).start()

    from gui.app import app
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)

if __name__ == "__main__":
    main()
