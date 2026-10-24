// Local device builds sign under a developer's own team and a separate bundle ID
// (see scripts/run-device.sh). Unset, the config is exactly app.json, so EAS
// production builds keep the App Store identifiers.
module.exports = ({ config }) => ({
  ...config,
  ios: {
    ...config.ios,
    bundleIdentifier: process.env.TARE_BUNDLE_ID || config.ios.bundleIdentifier,
    ...(process.env.TARE_APPLE_TEAM_ID ? { appleTeamId: process.env.TARE_APPLE_TEAM_ID } : {}),
  },
});
