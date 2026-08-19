#!/usr/bin/env bash
# Re-vendors the frontend's JS dependencies into web/vendor/ from npm, so
# the app has no CDN runtime dependency and works fully self-hosted/offline.
# Run this after bumping a version below.
set -euo pipefail

THREE_VERSION="0.160.0"
GSPLAT_VERSION="0.4.7"

cd "$(dirname "$0")/.."
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

(cd "$WORKDIR" && npm init -y >/dev/null 2>&1 && npm install --no-save \
  "three@${THREE_VERSION}" "@mkkellogg/gaussian-splats-3d@${GSPLAT_VERSION}")

rm -rf web/vendor
mkdir -p web/vendor/three/examples/jsm/controls web/vendor/three/examples/jsm/loaders web/vendor/gaussian-splats-3d

SRC="$WORKDIR/node_modules"
cp "$SRC/three/build/three.module.js" web/vendor/three/
cp "$SRC/three/examples/jsm/controls/OrbitControls.js" web/vendor/three/examples/jsm/controls/
cp "$SRC/three/examples/jsm/loaders/PLYLoader.js" web/vendor/three/examples/jsm/loaders/
cp "$SRC/three/examples/jsm/loaders/OBJLoader.js" web/vendor/three/examples/jsm/loaders/
cp "$SRC/@mkkellogg/gaussian-splats-3d/build/gaussian-splats-3d.module.js" web/vendor/gaussian-splats-3d/

echo "Vendored three@${THREE_VERSION} and gaussian-splats-3d@${GSPLAT_VERSION} into web/vendor/"
