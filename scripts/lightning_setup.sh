#!/usr/bin/env bash
# set up osmium on a lightning.ai studio (run once in the studio terminal).
#
# what it does:
#   1. installs uv if not present
#   2. clones the osmium repo into ~/osmium (or pulls latest if already cloned)
#   3. runs uv sync to install all python dependencies
#   4. prints a ready message with an example osmium train command
#
# if you forked the repo, update REPO_URL below.

set -euo pipefail

REPO_URL="git@github.com:williamangelo/osmium.git"
INSTALL_DIR="$HOME/osmium"

# 1. install uv if not already present
if ! command -v uv &>/dev/null; then
    echo "installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # add uv to PATH for the rest of this script
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "uv $(uv --version)"

# 2. clone or update repo
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "updating existing repo at $INSTALL_DIR..."
    git -C "$INSTALL_DIR" pull
else
    echo "cloning $REPO_URL into $INSTALL_DIR..."
    git clone "$REPO_URL" "$INSTALL_DIR"
fi

# 3. install python deps
echo "installing dependencies..."
cd "$INSTALL_DIR"
uv sync

echo ""
echo "osmium is ready. example training command:"
echo ""
echo "  cd $INSTALL_DIR"
echo "  uv run osmium train gpt2-small --data <dataset> --name <run-name>"
echo ""
echo "replace <dataset> with the name you used during preprocessing,"
echo "and <run-name> with a unique name for this training run."
