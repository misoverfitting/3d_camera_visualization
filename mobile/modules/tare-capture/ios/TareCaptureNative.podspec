Pod::Spec.new do |s|
  s.name = 'TareCaptureNative'
  s.version = '0.1.0'
  s.summary = 'Native ARKit RGB and pose capture for Tare'
  s.description = s.summary
  s.license = { :type => 'MIT' }
  s.author = 'Tare'
  s.homepage = 'https://github.com/misoverfitting/3d_camera_visualization'
  s.source = { :git => s.homepage }
  s.platform = :ios, '15.1'
  s.swift_version = '5.9'
  s.static_framework = true
  s.source_files = '**/*.swift'
  s.frameworks = 'ARKit', 'SceneKit', 'CoreImage', 'AVFoundation'
  s.dependency 'ExpoModulesCore'
  s.dependency 'ZIPFoundation', '~> 0.9.19'
end
