#import <Foundation/Foundation.h>
#import <CoreVideo/CoreVideo.h>
#import <simd/simd.h>

NS_ASSUME_NONNULL_BEGIN

/// Non-LiDAR depth: tracks image corners with pyramidal Lucas-Kanade flow and
/// triangulates them from known ARKit poses once a track has enough parallax.
/// Not thread-safe; call from one serial queue.
@interface TareFeatureTracker : NSObject

- (instancetype)initWithDownscale:(int)downscale NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;

/// `pixelBuffer` is ARFrame.capturedImage (bi-planar YCbCr; plane 0 is luma).
/// `intrinsics` refer to its full resolution; `cameraToWorld` is ARKit's camera transform.
/// Returns newly triangulated world points packed as float x,y,z triplets.
- (NSData *)processFrame:(CVPixelBufferRef)pixelBuffer
              intrinsics:(simd_float3x3)intrinsics
           cameraToWorld:(simd_float4x4)cameraToWorld;

/// Drops all tracks, e.g. after tracking loss when old observations no longer share a world frame.
- (void)reset;

@property (nonatomic, readonly) NSInteger activeTracks;
@property (nonatomic, readonly) NSInteger triangulatedTotal;

@end

NS_ASSUME_NONNULL_END
