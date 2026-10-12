#!/usr/bin/env bash
# ==============================================================================
# The actual training engine Anima Studio's Train tab drives ships bundled
# already at training_backend/ (see THIRD_PARTY_NOTICES.md) - nothing to
# clone here anymore. This just runs ITS installer, which pulls in PyTorch,
# xformers, and the rest of the ML training stack into its own virtual
# environment (separate from Anima Studio's own lightweight dependencies).
#
# Prefer doing this from the Train tab's "Set up backend" card instead - same
# installer, same output, but you don't need a terminal open for it. This
# script is here for anyone who'd rather run it directly.
# ==============================================================================
set -e
cd "$(dirname "$0")/training_backend"

if [ ! -f "install.sh" ]; then
    echo "training_backend/install.sh not found - is this a fresh checkout?"
    echo "If training_backend/ is missing entirely, redownload Anima Studio."
    exit 1
fi

echo "Running the bundled backend's installer. This downloads PyTorch and the"
echo "rest of the ML training stack, so it can take a while."
echo 'The first question it asks is "are you using this remotely? (y/n)" -'
echo 'answer n (you are running this locally, even though that phrasing makes'
echo "it easy to answer backwards). Answering y sends you down a different"
echo "path meant for remote/cloud setups (ngrok tunnels etc.) that Anima"
echo "Studio doesn't use."
echo

chmod +x install.sh
./install.sh

cat << 'EOF'

==============================================================================
Done (or the installer above hit a prompt/error - scroll up to check).

To start training from Anima Studio:
  1. Start the backend server - run training_backend/run.sh (Linux/Mac) to
     start it, or use the Train tab's "Launch backend" button instead.
  2. In Anima Studio's Train tab, set "Backend URL" to http://127.0.0.1:8000
     (the default) and click "Check connection".
  3. Fill in your settings and click "Start training".
==============================================================================
EOF
