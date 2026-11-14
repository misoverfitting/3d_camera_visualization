// Local device builds sign under a developer's own team and a separate bundle ID
// (see scripts/run-device.sh). Unset, the config is exactly app.json, so EAS
// production builds keep the App Store identifiers.
//
// TARE_API_URL / TARE_API_TOKEN point the app at the backend that proxies food
// identification to Claude (server/app/food_vision.py). A plain-http URL (a dev
// server on this network) also enables local-network loads.
const apiUrl = process.env.TARE_API_URL;
const localHttp = apiUrl && apiUrl.startsWith('http://');

module.exports = ({ config }) => ({
  ...config,
  ios: {
    ...config.ios,
    bundleIdentifier: process.env.TARE_BUNDLE_ID || config.ios.bundleIdentifier,
    ...(process.env.TARE_APPLE_TEAM_ID ? { appleTeamId: process.env.TARE_APPLE_TEAM_ID } : {}),
    infoPlist: {
      ...config.ios.infoPlist,
      ...(apiUrl ? { TareApiURL: apiUrl } : {}),
      ...(process.env.TARE_API_TOKEN ? { TareApiToken: process.env.TARE_API_TOKEN } : {}),
      ...(localHttp ? {
        NSAppTransportSecurity: { NSAllowsLocalNetworking: true },
        NSLocalNetworkUsageDescription: 'Reach the Tare development server on this network.',
      } : {}),
    },
  },
});
