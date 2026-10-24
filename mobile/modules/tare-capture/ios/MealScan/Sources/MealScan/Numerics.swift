import Foundation

enum Numerics {
  /// Solves A x = b (row-major n x n) by Gaussian elimination with partial pivoting.
  static func solve(_ a: [[Double]], _ b: [Double]) -> [Double]? {
    let n = b.count
    var m = a, rhs = b
    for column in 0..<n {
      guard let pivot = (column..<n).max(by: { abs(m[$0][column]) < abs(m[$1][column]) }),
            abs(m[pivot][column]) > 1e-15 else { return nil }
      m.swapAt(column, pivot); rhs.swapAt(column, pivot)
      for row in (column + 1)..<n {
        let factor = m[row][column] / m[column][column]
        if factor == 0 { continue }
        for c in column..<n { m[row][c] -= factor * m[column][c] }
        rhs[row] -= factor * rhs[column]
      }
    }
    var x = [Double](repeating: 0, count: n)
    for row in stride(from: n - 1, through: 0, by: -1) {
      var sum = rhs[row]
      for c in (row + 1)..<n { sum -= m[row][c] * x[c] }
      x[row] = sum / m[row][row]
    }
    return x
  }

  static func inverse(_ a: [[Double]]) -> [[Double]]? {
    let n = a.count
    var columns: [[Double]] = []
    for i in 0..<n {
      var e = [Double](repeating: 0, count: n); e[i] = 1
      guard let column = solve(a, e) else { return nil }
      columns.append(column)
    }
    return (0..<n).map { r in (0..<n).map { c in columns[c][r] } }
  }

  /// Levenberg-Marquardt on a residual function with a central-difference Jacobian.
  /// Returns the parameters and the Jacobian at the solution (for covariance).
  static func levenbergMarquardt(initial: [Double], steps: [Double], maxIterations: Int,
                                 residuals: ([Double]) -> [Double]?) -> (params: [Double], jacobian: [[Double]], residuals: [Double])? {
    var params = initial
    guard var current = residuals(params) else { return nil }
    var cost = current.reduce(0) { $0 + $1 * $1 }
    var lambda = 1e-3
    var jacobian: [[Double]] = []
    func computeJacobian(_ p: [Double]) -> [[Double]]? {
      var columns: [[Double]] = []
      for j in 0..<p.count {
        var plus = p, minus = p
        plus[j] += steps[j]; minus[j] -= steps[j]
        guard let rp = residuals(plus), let rm = residuals(minus), rp.count == rm.count else { return nil }
        columns.append(zip(rp, rm).map { ($0 - $1) / (2 * steps[j]) })
      }
      return (0..<columns[0].count).map { i in columns.map { $0[i] } }
    }
    for _ in 0..<maxIterations {
      guard let j = computeJacobian(params) else { return nil }
      jacobian = j
      let n = params.count
      var jtj = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
      var jtr = [Double](repeating: 0, count: n)
      for (row, r) in zip(j, current) {
        for a in 0..<n {
          jtr[a] += row[a] * r
          for b in 0..<n { jtj[a][b] += row[a] * row[b] }
        }
      }
      var improved = false
      while lambda < 1e12 {
        var damped = jtj
        for a in 0..<n { damped[a][a] += lambda * max(jtj[a][a], 1e-12) }
        guard let delta = solve(damped, jtr.map { -$0 }) else { lambda *= 10; continue }
        let candidate = zip(params, delta).map(+)
        if let r = residuals(candidate), r.count == current.count {
          let candidateCost = r.reduce(0) { $0 + $1 * $1 }
          if candidateCost < cost {
            let stepNorm = sqrt(delta.reduce(0) { $0 + $1 * $1 })
            params = candidate; current = r
            let converged = cost - candidateCost < 1e-18 + 1e-14 * cost || stepNorm < 1e-12
            cost = candidateCost; lambda = max(lambda / 10, 1e-12); improved = true
            if converged { return (params, computeJacobian(params) ?? jacobian, current) }
            break
          }
        }
        lambda *= 10
      }
      if !improved { break }
    }
    return (params, computeJacobian(params) ?? jacobian, current)
  }

  /// Linear-interpolated percentile of unsorted values, q in 0...1.
  static func percentile(_ values: [Double], _ q: Double) -> Double? {
    guard !values.isEmpty else { return nil }
    let sorted = values.sorted()
    let position = q * Double(sorted.count - 1)
    let lower = Int(position.rounded(.down)), upper = min(lower + 1, sorted.count - 1)
    return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - Double(lower))
  }
}

/// Deterministic generator so RANSAC results are reproducible in tests and evals.
public struct SplitMix64: RandomNumberGenerator {
  private var state: UInt64
  public init(seed: UInt64) { state = seed }
  public mutating func next() -> UInt64 {
    state &+= 0x9E3779B97F4A7C15
    var z = state
    z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
    return z ^ (z >> 31)
  }
}
