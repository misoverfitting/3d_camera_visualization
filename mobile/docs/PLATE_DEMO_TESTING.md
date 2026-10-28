# Testing the plate demo

How to build, run and check the plate demo on `feat/plate-demo`:
- tap a plate and circle it;
- the rim is found automatically;
- Claude labels each food item;
- 2–3 colored points land on each item.

It also covers the "Rim taps" test mode and the automated tests.

**You do not need real food.** See [What to put on the plate](#what-to-put-on-the-plate).

## Contents

1. [What you need](#what-you-need)
2. [Run the automated tests (no phone)](#run-the-automated-tests-no-phone)
3. [Start the backend](#start-the-backend)
4. [Check the backend without the app](#check-the-backend-without-the-app)
5. [Build and install the app](#build-and-install-the-app)
6. [What to put on the plate](#what-to-put-on-the-plate)
7. [Test the plate demo](#test-the-plate-demo)
8. [Test without LiDAR](#test-without-lidar)
9. [Test the Rim taps mode](#test-the-rim-taps-mode)
10. [Troubleshooting](#troubleshooting)
11. [Where things live](#where-things-live)

## What you need

- **Mac:** a current Xcode (tested with Xcode 27.1), Node 20+, Python 3.11+, and an Apple ID signed in under Xcode → Settings → Accounts. Without the account, builds fail with "No Accounts".
- **iPhone:** iOS 15.1 or newer, connected by USB or paired over Wi-Fi.
  - Use an iPhone Pro to see the LiDAR checks.
  - Any ARKit iPhone runs the demo itself.
  - Turn on Developer Mode: Settings → Privacy & Security → Developer Mode.
- **Anthropic API key:** an organization key with access to `claude-opus-5-5`.
- **Network:** the iPhone and the Mac on the same Wi-Fi, since the app calls a server on the Mac. A Railway deploy removes this requirement; see [Start the backend](#start-the-backend).
- **Something round and something to put on it** ([below](#what-to-put-on-the-plate)).

## Run the automated tests (no phone)

These need no device or key. Run them first.

```sh
# Swift pipeline core: rim geometry, edge rim fit, orbit selection, probe spots, food point lifting.
cd mobile/modules/tare-capture/ios/MealScan
swift test            # 27 tests, ~50 s in debug (rendered synthetic plates dominate)

# Backend food endpoint, with Claude stubbed out.
cd ../../../../../server
pip install fastapi httpx pytest python-multipart anthropic
python -m pytest tests/test_food_vision.py --noconftest -q   # 3 tests
```

The edge rim tests render a textured table with a plate (a darker inner well and a food blob) or a deep bowl, from 12 orbit views. They check that the fit finds the outer rim, not the well or the food, within 4 mm on:
- diameter;
- height;
- center position.

In practice the error is under 1 mm.

`--noconftest` skips `server/tests/conftest.py`, which imports the reconstruction pipeline and its heavy dependencies (open3d, pycolmap). Drop the flag if your environment already has `requirements.txt` installed.

## Start the backend

The phone never holds the Anthropic key. It posts a cropped JPEG to `POST /api/food/identify`, implemented in `server/app/food_vision.py`, and the server calls Claude.

### Option A: on your Mac (quickest)

```sh
cd server
pip install -r requirements.txt          # full server; or see the note below
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` is required so the phone can reach it.

**If `requirements.txt` is heavy for you** (open3d and pycolmap), you can serve just the food router instead:

```sh
pip install fastapi uvicorn anthropic python-multipart
python - <<'EOF'
import uvicorn
from fastapi import FastAPI
from app.food_vision import router
app = FastAPI(); app.include_router(router)
uvicorn.run(app, host="0.0.0.0", port=8000)
EOF
```

**Find the address the phone will use:**

```sh
scutil --get LocalHostName     # e.g. MacBook-Air-2  ->  http://MacBook-Air-2.local:8000
```

**Check that it's reachable from the network side:**

```sh
curl -s -o /dev/null -w '%{http_code}\n' http://MacBook-Air-2.local:8000/docs   # 200
```

If macOS asks whether Python may accept incoming connections, allow it.

### Option B: Railway (needed for a demo away from your Mac)

Deploy `server/` as usual and set these variables:

| Variable | Value |
| --- | --- |
| `ANTHROPIC_API_KEY` | the key |
| `TARE_API_TOKEN` | any long random string, so the endpoint isn't an open Claude proxy |
| `FOOD_VISION_EFFORT` *(optional)* | `low`, `medium` (default) or `high` |
| `FOOD_VISION_MODEL` *(optional)* | defaults to `claude-opus-5-5` |

When `TARE_API_TOKEN` is set, requests need the header `X-Tare-Token: <token>`, and the app sends it automatically if built with the same value (see [the next section](#build-and-install-the-app)).

## Check the backend without the app

Do this before involving the phone. Any overhead photo of a plate of food works: your camera roll, or an image search for "plate of food top down". Save it as `plate.jpg`.

```sh
curl -s -F image=@plate.jpg http://MacBook-Air-2.local:8000/api/food/identify | python3 -m json.tool
# Railway with a token:
curl -s -H "X-Tare-Token: $TARE_API_TOKEN" -F image=@plate.jpg https://<your-app>.up.railway.app/api/food/identify
```

**Expected response:** within a few seconds you get something like this. Coordinates are fractions of the image, with (0, 0) at the top-left.

```json
{
  "items": [
    {"label": "grilled chicken breast", "category": "protein", "confidence": 0.92,
     "box": [0.31, 0.22, 0.62, 0.51],
     "points": [[0.40, 0.31], [0.47, 0.36], [0.55, 0.42]]},
    {"label": "white rice", "category": "starch", "confidence": 0.95, "box": [...], "points": [[...], [...]]}
  ],
  "model": "claude-opus-5-5"
}
```

**Error codes:**

| Code | Meaning |
| --- | --- |
| 400 | Not a JPEG. Convert a PNG/HEIC first: `sips -s format jpeg in.heic --out plate.jpg`. |
| 401 | The server has `TARE_API_TOKEN` set and you didn't send the header. |
| 422 | The model declined the image. |
| 503 | The server has no valid `ANTHROPIC_API_KEY`. |

**To check point placement visually**, draw the points on the image:

```sh
pip install pillow
python3 - plate.jpg <<'EOF'
import json, sys, subprocess
from PIL import Image, ImageDraw
path = sys.argv[1]
out = json.loads(subprocess.check_output(["curl", "-s", "-F", f"image=@{path}", "http://MacBook-Air-2.local:8000/api/food/identify"]))
im = Image.open(path).convert("RGB"); d = ImageDraw.Draw(im); W, H = im.size
for i, item in enumerate(out["items"]):
    c = ["#E4572E", "#17BEBB", "#FFC914", "#76B041", "#B07CFF", "#FF6F91"][i % 6]
    x0, y0, x1, y1 = item["box"]; d.rectangle([x0*W, y0*H, x1*W, y1*H], outline=c, width=4)
    for x, y in item["points"]: d.ellipse([x*W-10, y*H-10, x*W+10, y*H+10], fill=c)
    d.text((x0*W+6, y0*H+6), item["label"], fill=c)
im.save("plate_labeled.jpg"); print("wrote plate_labeled.jpg")
EOF
open plate_labeled.jpg
```

Every point should sit clearly inside its item, not on an edge or on the plate.

## Build and install the app

From `mobile/`:

```sh
npm install                      # also fetches the pinned OpenCV iOS framework (postinstall)
xcrun devicectl list devices     # find your iPhone's UDID
```

```sh
TARE_API_URL=http://MacBook-Air-2.local:8000 \
TEAM_ID=<your Apple team ID> \
DEVICE=<iPhone UDID> \
BUNDLE_ID=com.tare.capture.<you> \
sh scripts/run-device.sh
```

- **`TEAM_ID`:** the 10-character ID shown in Xcode → Settings → Accounts → your team.
- **`BUNDLE_ID`:** pick one your team can register. `com.tare.capture` belongs to Ilya's team and `com.tare.capture.dev` to Morgan's.
- **`TARE_API_URL`:** baked into the app's Info.plist at build time. An `http://` URL also turns on local-network access (`NSAllowsLocalNetworking`). For Railway, use the `https://` URL and add `TARE_API_TOKEN=<token>`.

The script does the following:
1. Runs `expo prebuild --clean`.
2. Builds a self-contained **Release** app, so no Metro is needed.
3. Installs it and launches it.

The full build takes 5–10 minutes the first time.

**First launch:**
1. If iOS says the developer isn't trusted, go to Settings → General → VPN & Device Management → your Apple ID → **Trust**, then open **Tare Capture**. You can also run `xcrun devicectl device process launch --device <UDID> <bundle id>`.
2. Allow **Camera**.
3. Allow **Local Network** when it asks during the first food identification. If you tap Don't Allow, identification fails with a connection error; fix it in Settings → Tare Capture → Local Network.

## What to put on the plate

The rim fit only needs a round plate or bowl. Food labelling needs things that look like food from above. Claude labels what it sees, so non-food objects get labelled too, and that's still a useful test of point placement.

From best to worst:

1. **Anything edible you have:** fruit (an apple, banana slices, grapes), bread, crackers, cereal, nuts, a cookie, cheese, a spoon of peanut butter, coffee beans. Use 3–4 different things in separate piles. Three items in different colours is a great test.
2. **Packaged food, unopened:** a granola bar, a yogurt cup, a bag of chips. Claude labels the package ("granola bar"), and the points should land on it.
3. **A printed or on-screen photo of a plate of food, laid flat on a real plate:** the labels will be good. The 3D points land on the paper, which is fine because they sit on the plate surface anyway. An iPad showing a food photo works too, but the glare and its rectangular edge make the rim fit harder.
4. **Non-food stand-ins:** coloured paper cut into blobs, LEGO, a sponge, a folded napkin. Expect labels like "folded paper" or "yellow sponge", or an empty list. Use these only to check rim and point geometry.

**Setup tips:**
- **Plate:** a plate that contrasts with the table, like a white plate on a wooden or dark table. White on white, or a busy patterned tablecloth, is the hard case.
- **Spacing:** leave gaps between items so each gets its own label.
- **Lighting:** even light, avoiding hard shadows from a single lamp directly overhead.

## Test the plate demo

The app opens in **Plate demo** mode. The three buttons at the top switch between Plate demo, Rim taps and Scan.

### 1. Find the table

Point the phone down at the table about 40 cm above it and move slowly sideways. Cyan patches mark detected surfaces, and the message changes to "Tap the middle of the plate to start."

### 2. Start

Tap the **middle of the plate**. A yellow dot marks the tap. You can tap the middle again at any time during capture; taps are averaged and help the search lock on.

### 3. Circle the plate

Start roughly overhead, then circle while lowering the phone to about 30° above the table, keeping the plate near the middle of the screen. Move slowly: about one full circle in 20–30 seconds.

**On-screen guidance:**

| Message | What to do |
| --- | --- |
| "Slow down" | You're moving faster than 45°/s or 35 cm/s. |
| "Keep the plate in the middle" | The plate is drifting toward the edge of the frame. |
| "Raise the phone a little" | You're below 15° elevation. |
| "Stay 20–60 cm from the plate" | You're outside the accepted range (15 cm–1 m). |

**The panel shows:**
- views kept, up to 32;
- % of the hemisphere covered;
- tap count;
- "fitting…" while the rim is being fitted.

### 4. Watch the rim appear

After about 6 views a **white ring** appears on the rim and refines every few views.

**Check:**
- The ring sits on the **outer edge** of the plate, not the inner well and not a round food item.
- It stays put as you move, rather than sliding.

### 5. Finish

The scan finishes on its own at 60% coverage with at least 18 views, or press **Finish now** (needs at least 6 views). The ring turns **green**.

**The panel shows:**
- **Rim diameter and height**, from camera images only.
- **On LiDAR phones, a LiDAR cross-check line.** The diameter should be within about 5 mm and the height within a few mm. A larger gap means one of the two is wrong; note which by eye.
- **Ruler check:** measure the plate with a ruler or tape to the outer edge. The camera estimate should be within about 3%.

**Before food labelling finishes, you see the earlier probe spots:**
- 8 discs on a ring 1 inch (25.4 mm) inside the rim.
- **Orange** means the height was measured from nearby 3D points.
- **Red** means the height was copied from the other spots, or is an 8 mm guess when none have points.
- **"Points down/up 2 mm"** moves all of them together. Note how much you needed: a consistent offset points to a table-height bias we can fix in code.

### 6. Food labelling

Labelling starts automatically about a second after the ring turns green ("Identifying the food…"), and takes about 3–10 s.

**What happens:**
- The app uses the most overhead photo from the orbit (under 45° from straight down), cropped to the plate with 15% margin and rotated upright.
- The 8 ring spots are replaced by **2–3 coloured discs per item**.
- A **floating label** in the same colour sits about 3 cm above each item and turns to face you.
- The panel shows a **legend**: colour, label, food type, number of points, and "unsure" when Claude's confidence is under 0.6.

**Check:**
- Each item's discs sit **on that item**, not on the plate or a neighbouring item.
- Labels are plausible.
- Items touching each other are split sensibly.
- The **number of discs** may be lower than Claude returned. Points that would land off the plate are dropped, and an item with no remaining points disappears.

**Other controls:**
- **Identify again** re-runs labelling on the same photo and costs another call.
- **Restart** clears everything for a new plate.

### 7. Things worth trying

- Two plates on the table. Tap one; the other should be ignored.
- A bowl instead of a plate. The rim height should read 5–8 cm, not about 2 cm.
- Finish early with only overhead views. The rim height will be less certain; compare it with the LiDAR check.
- Background the app mid-scan. On return, tracking restarts and the demo resets to "Tap the middle of the plate".

## Test without LiDAR

To simulate a non-LiDAR iPhone on a Pro:
1. Switch to **Scan** mode and tap **"Force multi-view depth (benchmark)"**.
2. Switch back to **Plate demo**.
3. Run the demo again.

**Expected:**
- The rim fit is unchanged, because it never uses depth. Rim sizing working without LiDAR is the point of this approach.
- No LiDAR check line.
- Probe spots are mostly red (no depth-measured heights).
- Food points use the plate-surface plane instead of depth, so they sit at plate height rather than on top of tall food.

On an actual non-LiDAR iPhone, the toggle doesn't appear and the behaviour is the same.

## Test the Rim taps mode

This is a manual version of the rim measurement. It is useful to separate "is the rim maths right" from "did automatic edge-finding pick the right edge".

1. Switch to **Rim taps**. Make sure cyan surfaces are showing.
2. Hold the phone over the plate looking down and tap **Capture Overhead**. The photo opens.
3. Tap 8–12 points spread around the **outer edge of the rim**. Pinch to zoom for precision; **Undo** and **Clear** are at the bottom. Tap **Done**.
4. Step 15–30 cm to the side, lower the phone so the plate is seen at an angle, and tap **Capture Side**. Tap the rim again the same way.
5. Optionally type the true diameter in mm, then tap **Measure rim**.

**Results:**

| Line | What it shows |
| --- | --- |
| **Two-view** | Diameter ± uncertainty, rim height ± uncertainty, fit error in px, confidence. Fit error should be a few px. Above about 20 px means a tap mapping or orientation problem; please report it. |
| **Overhead only / Side only** | Diameter from one photo, assuming the rim is on the table and at the measured height. The "on table" number is noticeably larger; that's the height bias the two-view method removes. |
| **LiDAR reference** and **vs ruler** | Errors against depth and against your tape measure. |

Each measurement is saved to the phone. In the Files app, open On My iPhone → Tare Capture → `RimTests/<timestamp>/`. It contains both photos, `rim.json` with the camera positions, the taps and the results. These are exactly the eval fixtures we'll need later, so AirDrop a few back.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| Build: "No Accounts" / "No profiles for …" | Sign in under Xcode → Settings → Accounts, and use a `BUNDLE_ID` your team can register. |
| Launch: "invalid code signature … not explicitly trusted" | Trust the developer profile (Settings → General → VPN & Device Management). |
| Stuck on "Look down at the table…" | Needs a textured, lit surface. Move sideways rather than rotating in place. "Restart surface detection" is in Scan mode. |
| Ring locks onto the inner well or a round food | Tap the plate's middle again, and include lower side views. Report the plate type and table surface. |
| Ring never appears | Under 6 views kept. Watch the hint: usually "Slow down" or plate out of frame. |
| "Could not reach the server" | Wrong `TARE_API_URL`, phone on a different Wi-Fi, server bound to 127.0.0.1 instead of `0.0.0.0`, macOS firewall, or Local Network permission denied. Test from the phone's Safari: `http://<mac>.local:8000/docs` should load. |
| "Server error 503" | The server's `ANTHROPIC_API_KEY` is missing or invalid. |
| "Server error 401" | The server has `TARE_API_TOKEN` set but the app was built without the same `TARE_API_TOKEN`. |
| "No backend configured" | The app was built without `TARE_API_URL`. Rebuild with it. |
| "No overhead photo yet" | No view was within 45° of straight down. Restart and start from directly above. |
| "No food found on the plate" | Claude returned an empty list (common with non-food stand-ins). |
| Discs float above or sink into food | Use Points up/down and note the offset. Without LiDAR, discs sit at plate height by design. |

**Server logs:** each request appears in the uvicorn console. On Railway, check the service logs for `/api/food/identify`.

## Where things live

| Piece | File |
| --- | --- |
| Orbit keyframe selection, coverage | `mobile/modules/tare-capture/ios/MealScan/Sources/MealScan/Orbit.swift` |
| Automatic rim fit (image edges, no depth) | `…/MealScan/EdgeRimFitter.swift` |
| Ring of probe spots | `…/MealScan/ProbeRing.swift` |
| Crop mapping, lifting food points to 3D | `…/MealScan/FoodPoints.swift` |
| Tunables (all "initial guess") | `…/MealScan/PipelineConfig.swift` (`Orbit`, `EdgeRim`, `Probe`) |
| Demo flow, AR drawing | `mobile/modules/tare-capture/ios/PlateDemo.swift` |
| Overhead photo, upload, food points | `mobile/modules/tare-capture/ios/FoodID.swift` |
| Rim taps mode | `mobile/modules/tare-capture/ios/RimTapTest.swift`, `mobile/RimTest.tsx` |
| UI | `mobile/App.tsx` |
| Backend endpoint | `server/app/food_vision.py` (`POST /api/food/identify`) |
| Build-time backend config | `mobile/app.config.js` (`TARE_API_URL`, `TARE_API_TOKEN`) |
