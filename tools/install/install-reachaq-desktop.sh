#!/usr/bin/env bash

set -u

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DEFAULT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)
INSTALL_REPO=${REACHAQ_INSTALL_REPO:-$REPO_DEFAULT}
INSTALL_ENV=${REACHAQ_INSTALL_ENV:-reachaq}
INSTALL_CONFIG_DIR=${REACHAQ_INSTALL_CONFIG_DIR:-$HOME/Autotrainer}
SYSTEM_CONFIG=${REACHAQ_SYSTEM_CONFIG:-$INSTALL_CONFIG_DIR/system_configuration.yaml}
USER_CONFIG_DIR=${REACHAQ_USER_CONFIG_DIR:-$HOME/.config/reachaq}
USER_DATA_DIR=${REACHAQ_USER_DATA_DIR:-$HOME/.local/share}
USER_BIN_DIR=${REACHAQ_USER_BIN_DIR:-$HOME/.local/bin}
DESKTOP_DIR=${REACHAQ_DESKTOP_DIR:-}
RUNTIME_ENV=${REACHAQ_RUNTIME_ENV:-$USER_CONFIG_DIR/runtime.env}

if [ "$#" -ne 0 ]; then
    printf 'install-reachaq-desktop.sh does not accept arguments.\n' >&2
    exit 2
fi

find_conda() {
    if [ -n "${REACHAQ_CONDA_BIN:-}" ] && [ -x "$REACHAQ_CONDA_BIN" ]; then
        printf '%s\n' "$REACHAQ_CONDA_BIN"
        return
    fi
    if command -v conda >/dev/null 2>&1; then
        command -v conda
        return
    fi
    local candidate
    for candidate in \
        "$HOME/anaconda3/bin/conda" \
        "$HOME/miniconda3/bin/conda" \
        "$HOME/mambaforge/bin/conda"; do
        if [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return
        fi
    done
    return 1
}

if [ -z "$DESKTOP_DIR" ]; then
    DESKTOP_DIR=$(xdg-user-dir DESKTOP 2>/dev/null || true)
    DESKTOP_DIR=${DESKTOP_DIR:-$HOME/Desktop}
fi
CONDA_BIN=$(find_conda) || {
    printf 'Unable to locate a Conda executable for the reachAQ launcher.\n' >&2
    exit 3
}

for path_value in "$CONDA_BIN" "$INSTALL_REPO" "$SYSTEM_CONFIG" "$RUNTIME_ENV"; do
    case "$path_value" in
        *$'\n'*|*$'\r'*)
            printf 'Launcher paths cannot contain newlines.\n' >&2
            exit 4
            ;;
    esac
done

mkdir -p \
    "$USER_BIN_DIR" \
    "$USER_CONFIG_DIR" \
    "$USER_DATA_DIR/applications" \
    "$USER_DATA_DIR/icons/hicolor/192x192/apps" \
    "$DESKTOP_DIR"

install -m 0755 "$SCRIPT_DIR/desktop/reachaq-launcher" \
    "$USER_BIN_DIR/reachaq-launcher"
install -m 0644 "$INSTALL_REPO/tools/acquisition/view/autotrainer.png" \
    "$USER_DATA_DIR/icons/hicolor/192x192/apps/reachaq.png"

config_tmp=$(mktemp "$USER_CONFIG_DIR/launcher.conf.XXXXXX") || exit 5
desktop_tmp=$(mktemp "$USER_CONFIG_DIR/reachaq.desktop.XXXXXX") || {
    rm -f "$config_tmp"
    exit 5
}
trap 'rm -f "$config_tmp" "$desktop_tmp"' EXIT
printf '%s\n' \
    "CONDA_BIN=$CONDA_BIN" \
    "CONDA_ENV=$INSTALL_ENV" \
    "REPOSITORY=$INSTALL_REPO" \
    "SYSTEM_CONFIG=$SYSTEM_CONFIG" \
    "RUNTIME_ENV=$RUNTIME_ENV" > "$config_tmp"
chmod 0600 "$config_tmp"
mv -f "$config_tmp" "$USER_CONFIG_DIR/launcher.conf"

escaped_exec=$(printf '%s' "$USER_BIN_DIR/reachaq-launcher" | sed 's/[&|]/\\&/g')
sed "s|@EXEC@|$escaped_exec|g" \
    "$SCRIPT_DIR/desktop/reachaq.desktop.in" > "$desktop_tmp"
install -m 0755 "$desktop_tmp" "$USER_DATA_DIR/applications/reachaq.desktop"
install -m 0755 "$desktop_tmp" "$DESKTOP_DIR/reachAQ.desktop"

# Replace the obsolete non-executable shell fragment from older installations.
if [ -f "$DESKTOP_DIR/ReachAQ-startup" ] || [ -L "$DESKTOP_DIR/ReachAQ-startup" ]; then
    rm -f "$DESKTOP_DIR/ReachAQ-startup"
fi

if command -v desktop-file-validate >/dev/null 2>&1; then
    desktop-file-validate "$USER_DATA_DIR/applications/reachaq.desktop"
fi
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$USER_DATA_DIR/applications" >/dev/null 2>&1 || true
fi
if command -v gio >/dev/null 2>&1; then
    gio set "$DESKTOP_DIR/reachAQ.desktop" metadata::trusted true >/dev/null 2>&1 || true
fi

printf 'Installed terminal-visible reachAQ launcher: %s\n' \
    "$DESKTOP_DIR/reachAQ.desktop"
