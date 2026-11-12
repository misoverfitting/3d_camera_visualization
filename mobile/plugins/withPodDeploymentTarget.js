// Raises pod targets below the app's minimum iOS version. Recent Xcode releases
// reject the iOS 12 target that ZIPFoundation's resource bundle still declares.
const fs = require('fs');
const path = require('path');
const { withDangerousMod } = require('expo/config-plugins');

const MINIMUM = '15.1';
const MARKER = '# tare: pod deployment target';

module.exports = function withPodDeploymentTarget(config) {
  return withDangerousMod(config, ['ios', (config) => {
    const podfile = path.join(config.modRequest.platformProjectRoot, 'Podfile');
    let contents = fs.readFileSync(podfile, 'utf8');
    if (!contents.includes(MARKER)) {
      contents = contents.replace('post_install do |installer|', `post_install do |installer|
    ${MARKER}
    installer.pods_project.targets.each do |target|
      target.build_configurations.each do |build|
        if build.build_settings['IPHONEOS_DEPLOYMENT_TARGET'].to_f < ${MINIMUM}
          build.build_settings['IPHONEOS_DEPLOYMENT_TARGET'] = '${MINIMUM}'
        end
      end
    end`);
      fs.writeFileSync(podfile, contents);
    }
    return config;
  }]);
};
