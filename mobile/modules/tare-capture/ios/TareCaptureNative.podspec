Pod::Spec.new do |s|
  s.name = 'TareCaptureNative'
  s.version = '0.1.0'
  s.summary = 'Native ARKit capture with LiDAR and multi-view depth for Tare'
  s.description = s.summary
  s.license = { :type => 'MIT' }
  s.author = 'Tare'
  s.homepage = 'https://github.com/misoverfitting/3d_camera_visualization'
  s.source = { :git => s.homepage }
  s.platform = :ios, '15.1'
  s.swift_version = '5.9'
  s.static_framework = true
  # Top level only (vendor/ holds OpenCV's own headers), plus the pure-Swift
  # MealScan pipeline core, which is also a standalone package for `swift test`.
  s.source_files = '*.{h,mm,swift}', 'MealScan/Sources/**/*.swift'
  s.public_header_files = 'TareFeatureTracker.h'
  # Fetched by mobile/scripts/fetch-opencv.sh (npm postinstall); not committed.
  s.vendored_frameworks = 'vendor/opencv2.framework'
  s.frameworks = 'ARKit', 'SceneKit', 'CoreImage', 'AVFoundation', 'Accelerate', 'CoreMedia', 'CoreVideo'
  s.libraries = 'c++', 'z'
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    # OpenCV's headers trip doc-comment warnings that are not ours to fix.
    'CLANG_WARN_DOCUMENTATION_COMMENTS' => 'NO',
  }
  s.dependency 'ExpoModulesCore'
  s.dependency 'ZIPFoundation', '~> 0.9.19'
end
