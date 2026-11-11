import Foundation
import simd

/// Minimal binary glTF 2.0 (.glb) reader: node tree, triangle meshes (positions, normals,
/// indices) and PBR material factors. Enough for the probe prototype model (a three.js
/// export); textures, skins, morph targets and compression extensions are not supported.
public struct GLBModel: Sendable {
  public struct Material: Sendable {
    public var name: String
    /// Linear RGBA.
    public var baseColor: SIMD4<Double>
    public var metallic: Double
    public var roughness: Double
    /// Linear RGB, already multiplied by KHR_materials_emissive_strength.
    public var emissive: SIMD3<Double>
  }

  public struct Primitive: Sendable {
    public var positions: [SIMD3<Float>]
    public var normals: [SIMD3<Float>]
    public var indices: [UInt32]
    public var material: Int?
  }

  public struct Node: Sendable {
    public var name: String
    /// Parent-from-node transform.
    public var matrix: simd_double4x4
    public var primitives: [Primitive]
    public var children: [Int]
  }

  public enum Failure: Error, Equatable {
    case notGLB, badChunk, badJSON, unsupported(String), outOfBounds
  }

  public var nodes: [Node]
  public var roots: [Int]
  public var materials: [Material]

  public init(data: Data) throws {
    let bytes = [UInt8](data)
    func u32(_ at: Int) throws -> Int {
      guard at + 4 <= bytes.count else { throw Failure.outOfBounds }
      return Int(UInt32(bytes[at]) | UInt32(bytes[at + 1]) << 8 | UInt32(bytes[at + 2]) << 16 | UInt32(bytes[at + 3]) << 24)
    }
    guard bytes.count >= 20, try u32(0) == 0x4654_6C67, try u32(4) == 2 else { throw Failure.notGLB }
    var json: [String: Any]?
    var bin: ArraySlice<UInt8> = []
    var offset = 12
    while offset + 8 <= bytes.count {
      let length = try u32(offset), type = try u32(offset + 4)
      guard offset + 8 + length <= bytes.count else { throw Failure.badChunk }
      let chunk = bytes[(offset + 8)..<(offset + 8 + length)]
      if type == 0x4E4F_534A {
        json = try JSONSerialization.jsonObject(with: Data(chunk)) as? [String: Any]
      } else if type == 0x004E_4942 {
        bin = chunk
      }
      offset += 8 + length
    }
    guard let json else { throw Failure.badJSON }
    if let required = json["extensionsRequired"] as? [String], let first = required.first { throw Failure.unsupported(first) }

    let views = json["bufferViews"] as? [[String: Any]] ?? []
    let accessors = json["accessors"] as? [[String: Any]] ?? []
    let binStart = bin.startIndex

    /// Reads an accessor as rows of `Double` components.
    func read(_ index: Int) throws -> [[Double]] {
      guard index < accessors.count else { throw Failure.outOfBounds }
      let accessor = accessors[index]
      guard let viewIndex = accessor["bufferView"] as? Int, viewIndex < views.count else { throw Failure.unsupported("sparse accessor") }
      let view = views[viewIndex]
      let componentType = accessor["componentType"] as? Int ?? 0
      let count = accessor["count"] as? Int ?? 0
      let width = ["SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4][accessor["type"] as? String ?? ""] ?? 0
      let size = [5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4][componentType] ?? 0
      guard width > 0, size > 0 else { throw Failure.unsupported("accessor type") }
      let stride = (view["byteStride"] as? Int) ?? width * size
      let base = binStart + (view["byteOffset"] as? Int ?? 0) + (accessor["byteOffset"] as? Int ?? 0)
      guard count == 0 || base + (count - 1) * stride + width * size <= bin.endIndex else { throw Failure.outOfBounds }
      var rows: [[Double]] = []
      rows.reserveCapacity(count)
      for i in 0..<count {
        var row = [Double](repeating: 0, count: width)
        for c in 0..<width {
          let at = base + i * stride + c * size
          switch componentType {
          case 5126:
            let raw = UInt32(bin[at]) | UInt32(bin[at + 1]) << 8 | UInt32(bin[at + 2]) << 16 | UInt32(bin[at + 3]) << 24
            row[c] = Double(Float(bitPattern: raw))
          case 5125: row[c] = Double(UInt32(bin[at]) | UInt32(bin[at + 1]) << 8 | UInt32(bin[at + 2]) << 16 | UInt32(bin[at + 3]) << 24)
          case 5123: row[c] = Double(UInt16(bin[at]) | UInt16(bin[at + 1]) << 8)
          case 5122: row[c] = Double(Int16(bitPattern: UInt16(bin[at]) | UInt16(bin[at + 1]) << 8))
          case 5121: row[c] = Double(bin[at])
          default: row[c] = Double(Int8(bitPattern: bin[at]))
          }
        }
        rows.append(row)
      }
      return rows
    }

    materials = (json["materials"] as? [[String: Any]] ?? []).map { m in
      let pbr = m["pbrMetallicRoughness"] as? [String: Any] ?? [:]
      let color = pbr["baseColorFactor"] as? [Double] ?? [1, 1, 1, 1]
      let emissive = m["emissiveFactor"] as? [Double] ?? [0, 0, 0]
      let strength = ((m["extensions"] as? [String: Any])?["KHR_materials_emissive_strength"] as? [String: Any])?["emissiveStrength"] as? Double ?? 1
      return Material(name: m["name"] as? String ?? "",
                      baseColor: color.count == 4 ? SIMD4(color[0], color[1], color[2], color[3]) : SIMD4(1, 1, 1, 1),
                      metallic: pbr["metallicFactor"] as? Double ?? 1, roughness: pbr["roughnessFactor"] as? Double ?? 1,
                      emissive: emissive.count == 3 ? strength * SIMD3(emissive[0], emissive[1], emissive[2]) : .zero)
    }

    let meshes: [[Primitive]] = try (json["meshes"] as? [[String: Any]] ?? []).map { mesh in
      try (mesh["primitives"] as? [[String: Any]] ?? []).compactMap { primitive -> Primitive? in
        guard (primitive["mode"] as? Int ?? 4) == 4 else { return nil }  // Triangles only.
        let attributes = primitive["attributes"] as? [String: Any] ?? [:]
        guard let positionIndex = attributes["POSITION"] as? Int else { return nil }
        let positions = try read(positionIndex).map { SIMD3<Float>(Float($0[0]), Float($0[1]), Float($0[2])) }
        let normals = try (attributes["NORMAL"] as? Int).map { try read($0).map { SIMD3<Float>(Float($0[0]), Float($0[1]), Float($0[2])) } } ?? []
        let indices = try (primitive["indices"] as? Int).map { try read($0).map { UInt32($0[0]) } }
          ?? (0..<UInt32(positions.count)).map { $0 }
        guard indices.allSatisfy({ Int($0) < positions.count }) else { throw Failure.outOfBounds }
        return Primitive(positions: positions, normals: normals.count == positions.count ? normals : [],
                         indices: indices, material: primitive["material"] as? Int)
      }
    }

    nodes = try (json["nodes"] as? [[String: Any]] ?? []).map { node in
      var matrix = matrix_identity_double4x4
      if let m = node["matrix"] as? [Double], m.count == 16 {
        matrix = simd_double4x4(SIMD4(m[0], m[1], m[2], m[3]), SIMD4(m[4], m[5], m[6], m[7]),
                                SIMD4(m[8], m[9], m[10], m[11]), SIMD4(m[12], m[13], m[14], m[15]))
      } else {
        let t = node["translation"] as? [Double] ?? [0, 0, 0]
        let r = node["rotation"] as? [Double] ?? [0, 0, 0, 1]
        let s = node["scale"] as? [Double] ?? [1, 1, 1]
        let rotation = simd_double3x3(simd_quatd(ix: r[0], iy: r[1], iz: r[2], r: r[3]))
        matrix = simd_double4x4(SIMD4(s[0] * rotation.columns.0, 0), SIMD4(s[1] * rotation.columns.1, 0),
                                SIMD4(s[2] * rotation.columns.2, 0), SIMD4(t[0], t[1], t[2], 1))
      }
      let mesh = node["mesh"] as? Int
      guard mesh.map({ $0 < meshes.count }) ?? true else { throw Failure.outOfBounds }
      return Node(name: node["name"] as? String ?? "", matrix: matrix, primitives: mesh.map { meshes[$0] } ?? [],
                  children: node["children"] as? [Int] ?? [])
    }
    let sceneIndex = json["scene"] as? Int ?? 0
    let scenes = json["scenes"] as? [[String: Any]] ?? []
    roots = sceneIndex < scenes.count ? scenes[sceneIndex]["nodes"] as? [Int] ?? [] : []
    let count = nodes.count
    guard roots.allSatisfy({ $0 < count }), nodes.allSatisfy({ $0.children.allSatisfy { $0 < count } }) else { throw Failure.outOfBounds }
  }

  public func index(named name: String) -> Int? { nodes.firstIndex { $0.name == name } }

  /// Model-from-node transforms for every node reachable from the scene roots.
  public func modelMatrices() -> [Int: simd_double4x4] {
    var out: [Int: simd_double4x4] = [:]
    func walk(_ index: Int, _ parent: simd_double4x4) {
      guard out[index] == nil else { return }  // Guards against cycles in malformed files.
      let matrix = parent * nodes[index].matrix
      out[index] = matrix
      nodes[index].children.forEach { walk($0, matrix) }
    }
    roots.forEach { walk($0, matrix_identity_double4x4) }
    return out
  }

  /// Vertex positions of a node's own meshes, in model coordinates.
  public func modelPositions(of index: Int) -> [SIMD3<Double>] {
    guard let matrix = modelMatrices()[index] else { return [] }
    return nodes[index].primitives.flatMap(\.positions).map { p in
      let q = matrix * SIMD4(Double(p.x), Double(p.y), Double(p.z), 1)
      return SIMD3(q.x, q.y, q.z)
    }
  }
}
