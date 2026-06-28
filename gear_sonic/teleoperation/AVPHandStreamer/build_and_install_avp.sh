#!/usr/bin/env bash
# build_and_install_avp.sh
# Builds AVPHandStreamer and installs it on a connected Apple Vision Pro.
# Usage: bash build_and_install_avp.sh [TEAM_ID]
#   TEAM_ID  Your Apple Developer Team ID (e.g. ABCD1234EF).
#            Looks it up automatically if omitted.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$SCRIPT_DIR/AVPHandStreamer.xcodeproj"
SCHEME="AVPHandStreamer"
ARCHIVE="$SCRIPT_DIR/build/AVPHandStreamer.xcarchive"

# ── 0. Verify Xcode is active ─────────────────────────────────────────────────
if ! command -v xcodebuild &>/dev/null; then
    echo "[ERROR] xcodebuild not found. Install Xcode from the App Store, then run:"
    echo "        sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"
    exit 1
fi
echo "[OK] $(xcodebuild -version | head -1)"

# ── 1. Resolve Team ID ────────────────────────────────────────────────────────
TEAM_ID="${1:-}"
# First try: read from the project file (most reliable — set via Xcode UI)
if [[ -z "$TEAM_ID" ]]; then
    TEAM_ID="$(grep -m1 'DEVELOPMENT_TEAM = ' "$PROJECT/project.pbxproj" \
               | sed 's/.*DEVELOPMENT_TEAM = \([^;]*\);.*/\1/' | tr -d '[:space:]' || true)"
fi
# Second try: keychain certificate
if [[ -z "$TEAM_ID" ]]; then
    TEAM_ID="$(security find-certificate -a -c 'Apple Development' \
                 -p ~/Library/Keychains/login.keychain-db 2>/dev/null \
               | openssl x509 -noout -subject 2>/dev/null \
               | grep -oE 'OU=[A-Z0-9]+' | head -1 | cut -d= -f2 || true)"
fi
if [[ -z "$TEAM_ID" ]]; then
    echo "[WARN] Could not detect Team ID automatically."
    echo "       Open AVPHandStreamer.xcodeproj in Xcode and set your team in"
    echo "       Signing & Capabilities, then re-run this script."
    TEAM_ARG=""
else
    echo "[OK] Using Team ID: $TEAM_ID"
    TEAM_ARG="DEVELOPMENT_TEAM=$TEAM_ID"
fi

# ── 2. Detect connected Apple Vision Pro ─────────────────────────────────────
echo "[INFO] Looking for connected Apple Vision Pro..."
DEVICE_ID="$(xcrun devicectl list devices 2>/dev/null \
             | grep -i "apple vision" \
             | grep -oE '[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}' \
             | head -1 || true)"

if [[ -z "$DEVICE_ID" ]]; then
    echo "[WARN] No Apple Vision Pro found via USB. Will build and archive only."
    echo "       Connect AVP via USB-C or pair wirelessly in Xcode → Window → Devices."
    INSTALL=false
else
    echo "[OK] Found AVP: $DEVICE_ID"
    INSTALL=true
fi

# ── 3. Build archive ──────────────────────────────────────────────────────────
echo "[INFO] Building archive..."
mkdir -p "$SCRIPT_DIR/build"
xcodebuild \
    -project "$PROJECT" \
    -scheme "$SCHEME" \
    -destination "generic/platform=visionOS" \
    -sdk xros \
    -archivePath "$ARCHIVE" \
    ${TEAM_ARG:+DEVELOPMENT_TEAM="${TEAM_ID}"} \
    CODE_SIGN_STYLE=Automatic \
    -allowProvisioningUpdates \
    clean archive

echo "[OK] Archive: $ARCHIVE"

# ── 4. Export IPA ─────────────────────────────────────────────────────────────
EXPORT_DIR="$SCRIPT_DIR/build/export"
EXPORT_PLIST="$SCRIPT_DIR/build/ExportOptions.plist"
cat > "$EXPORT_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>method</key>
    <string>development</string>
    <key>destination</key>
    <string>export</string>
    <key>signingStyle</key>
    <string>automatic</string>
    ${TEAM_ID:+<key>teamID</key><string>$TEAM_ID</string>}
</dict>
</plist>
PLIST

xcodebuild -exportArchive \
    -archivePath "$ARCHIVE" \
    -exportPath "$EXPORT_DIR" \
    -exportOptionsPlist "$EXPORT_PLIST"

IPA="$(find "$EXPORT_DIR" -name "*.ipa" | head -1)"
echo "[OK] IPA: $IPA"

# ── 5. Install on device (if found) ──────────────────────────────────────────
if [[ "$INSTALL" == true && -n "$IPA" ]]; then
    echo "[INFO] Installing on AVP ($DEVICE_ID)..."
    xcrun devicectl device install app \
        --device "$DEVICE_ID" \
        "$IPA"
    echo "[OK] Installed! On AVP: trust the developer under Settings → General → VPN & Device Management."
else
    echo "[INFO] To install manually later:"
    echo "       xcrun devicectl device install app --device <DEVICE_ID> \"$IPA\""
    echo "       (find DEVICE_ID with: xcrun devicectl list devices)"
fi
