#!/bin/bash
#
# Script to open TGLite documentation using detected browser.
#

# Get the directory where the script is located.
SCRIPT_DIR="$(dirname "$0")"

# Path to the HTML file you want to open.
HTML_PATH="$SCRIPT_DIR/build/html/index.html"

# Function to open the HTML file using a browser.
open_html() {
    local path=$1
    if command -v xdg-open &> /dev/null; then
        # Preferred way to open files on Desktop Linux
        xdg-open "$path"
    elif command -v gnome-open &> /dev/null; then
        # For systems with Gnome.
        gnome-open "$path"
    elif command -v x-www-browser &> /dev/null; then
        # A generic way to open files, might work when xdg-open and gnome-open are unavailable.
        x-www-browser "$path"
    else
        echo "Could not detect the web browser to open the documentation."
        return 1
    fi
}

# Detect the operating system and open the HTML file.
case "$(uname -s)" in
    Linux*)
        open_html "$HTML_PATH"
        ;;
    Darwin*)
        open "$HTML_PATH"
        ;;
    CYGWIN*|MINGW32*|MSYS*|MINGW*)
        start "$HTML_PATH"
        ;;
    *)
        echo "Unknown operating system. Cannot open the documentation automatically."
        ;;
esac
