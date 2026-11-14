// OpenCV is an iOS build dependency. Windows development uses the EAS macOS builder.
const { spawnSync } = require('node:child_process');
const path = require('node:path');
if (process.platform === 'win32') {
  console.log('Skipping iOS OpenCV download on Windows; EAS downloads it during the native build.');
} else {
  const result = spawnSync('sh', [path.join(__dirname, 'fetch-opencv.sh')], { stdio: 'inherit' });
  if (result.error) throw result.error;
  process.exit(result.status ?? 1);
}
