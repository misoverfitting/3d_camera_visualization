// plateitems <image>... --out DIR
// For each image writes DIR/<stem>.json (regions: box, points, area, in normalized
// coordinates) and DIR/<stem>.png (region k+1 per pixel, 0 = none) at analysis size.
import AppKit
import CoreGraphics
import Foundation
import ImageIO
import MealScan
import Vision

let analysisSide = 256.0

func rgba(_ image: CGImage, width w: Int, height h: Int) -> [UInt8] {
  var data = [UInt8](repeating: 0, count: w * h * 4)
  let context = CGContext(data: &data, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                          space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
  context.interpolationQuality = .medium
  context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
  return data
}

/// The plate (with its food) as Vision's foreground mask, at w x h.
func plateMask(_ image: CGImage, width w: Int, height h: Int) -> [Bool]? {
  let handler = VNImageRequestHandler(cgImage: image)
  let request = VNGenerateForegroundInstanceMaskRequest()
  guard (try? handler.perform([request])) != nil, let observation = request.results?.first,
        let buffer = try? observation.generateScaledMaskForImage(forInstances: observation.allInstances, from: handler) else { return nil }
  CVPixelBufferLockBaseAddress(buffer, .readOnly)
  defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
  let bw = CVPixelBufferGetWidth(buffer), bh = CVPixelBufferGetHeight(buffer), stride = CVPixelBufferGetBytesPerRow(buffer) / 4
  let p = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: Float32.self)
  return (0..<(w * h)).map { i in
    let x = min(bw - 1, (i % w) * bw / w), y = min(bh - 1, (i / w) * bh / h)
    return p[y * stride + x] > 0.5
  }
}

func erode(_ m: [Bool], _ w: Int, _ h: Int, _ n: Int) -> [Bool] {
  var out = m
  for _ in 0..<n {
    let src = out
    for y in 0..<h { for x in 0..<w where src[y * w + x] {
      if x == 0 || y == 0 || x == w - 1 || y == h - 1 || !src[y * w + x - 1] || !src[y * w + x + 1] || !src[(y - 1) * w + x] || !src[(y + 1) * w + x] {
        out[y * w + x] = false
      }
    } }
  }
  return out
}

func writeIndexPNG(_ index: [UInt8], _ w: Int, _ h: Int, to url: URL) {
  var data = index
  let context = CGContext(data: &data, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w,
                          space: CGColorSpaceCreateDeviceGray(), bitmapInfo: 0)!
  let image = context.makeImage()!
  let destination = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil)!
  CGImageDestinationAddImage(destination, image, nil)
  CGImageDestinationFinalize(destination)
}

var args = Array(CommandLine.arguments.dropFirst())
guard let outFlag = args.firstIndex(of: "--out"), outFlag + 1 < args.count else {
  FileHandle.standardError.write(Data("usage: plateitems <image>... --out DIR\n".utf8)); exit(2)
}
let out = URL(fileURLWithPath: args[outFlag + 1])
args.removeSubrange(outFlag...(outFlag + 1))
try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)

var done = 0
for path in args {
  let url = URL(fileURLWithPath: path)
  // Nutrition5k folders hold <dish>/rgb.png: name the output after the dish.
  let base = url.deletingPathExtension().lastPathComponent
  let stem = base == "rgb" ? url.deletingLastPathComponent().lastPathComponent : base
  let jsonURL = out.appendingPathComponent("\(stem).json")
  if FileManager.default.fileExists(atPath: jsonURL.path) { done += 1; continue }
  guard let source = CGImageSourceCreateWithURL(url as CFURL, nil), let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { continue }
  let scale = min(1, analysisSide / Double(max(image.width, image.height)))
  let w = Int((Double(image.width) * scale).rounded()), h = Int((Double(image.height) * scale).rounded())
  autoreleasepool {
    let interior = plateMask(image, width: w, height: h) ?? [Bool](repeating: true, count: w * h)
    let inner = erode(interior, w, h, Int(Double(min(w, h)) * 0.06))
    let ring = zip(interior, inner).map { $0 && !$1 }
    let regions = PlateItems.segment(RGBImage(widthPx: w, heightPx: h, rgba: rgba(image, width: w, height: h)), interior: interior, ring: ring)
    var index = [UInt8](repeating: 0, count: w * h)
    for (k, region) in regions.prefix(254).enumerated() { for i in 0..<(w * h) where region.mask[i] { index[i] = UInt8(k + 1) } }
    writeIndexPNG(index, w, h, to: out.appendingPathComponent("\(stem).png"))
    let json: [String: Any] = ["image": path, "width": w, "height": h, "regions": regions.prefix(254).map { r in
      ["box": [r.box.x, r.box.y, r.box.z, r.box.w], "area": r.areaPx, "points": r.points.map { [$0.x, $0.y] }]
    }]
    try? JSONSerialization.data(withJSONObject: json).write(to: jsonURL)
  }
  done += 1
  if done % 200 == 0 { print("\(done)/\(args.count)"); fflush(stdout) }
}
print("done \(done)")
