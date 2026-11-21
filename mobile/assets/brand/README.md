# TARE mobile brand assets

Source: https://github.com/hn-xyn/tare-website at
`c2cdfa3fc95f2bddc54bac450a6c486b91ab7ab1` (2026-11-14).

- Wordmark: `INTERIM/public/assets/tare-lockup-ink.svg`.
- Horizon photograph: `INTERIM/public/assets/plates/plate-horizon-glow-2.jpg`.
- Icon: `INTERIM/public/favicon.svg` and `INTERIM/favicon-tare/favicon-512.png`.
  `icon.png` composites the supplied transparent PNG on the site's warm gray and
  resizes to the 1024px opaque iOS icon format. The source art is unchanged.
- Colors: `INTERIM/public/styles.css`: background `#D3D1CD`, ink `#1A1A18`,
  body `#3A3A37`, orange `#FF6A00`, ink rules at 22–28% opacity.
- Typography: Instrument Sans and Sometype Mono, as used in the public site's
  `index.html`. TTFs are bundled for offline use, from the Google Fonts `ofl`
  repository. The accompanying SIL Open Font Licenses are included here.

`Brand.tsx` embeds the wordmark paths without the SVG's provenance metadata for
React Native SVG rendering. The source SVG is retained here. The app uses the
site's small square markers, thin rules and restrained controls across all modes.
Food-region colors retain distinct identities so probe points remain distinguishable.
