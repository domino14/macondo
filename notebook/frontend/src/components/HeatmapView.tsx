import type { HeatmapData } from '../types'
import Board from './Board'

// Color scale: transparent → blue → cyan → green → yellow → red
function heatColor(v: number): string {
  if (v <= 0) return 'transparent'

  // Map 0..1 through a 4-stop gradient
  // 0.0 → blue   (#3b82f6)
  // 0.33 → cyan  (#06b6d4)
  // 0.66 → yellow (#f59e0b)
  // 1.0 → red    (#ef4444)
  const stops: [number, [number, number, number]][] = [
    [0.0,  [59,  130, 246]],  // blue
    [0.33, [6,   182, 212]],  // cyan
    [0.66, [245, 158, 11]],   // yellow
    [1.0,  [239, 68,  68]],   // red
  ]

  let lo = stops[0], hi = stops[stops.length - 1]
  for (let i = 0; i < stops.length - 1; i++) {
    if (v >= stops[i][0] && v <= stops[i + 1][0]) {
      lo = stops[i]
      hi = stops[i + 1]
      break
    }
  }

  const t = (v - lo[0]) / (hi[0] - lo[0])
  const r = Math.round(lo[1][0] + t * (hi[1][0] - lo[1][0]))
  const g = Math.round(lo[1][1] + t * (hi[1][1] - lo[1][1]))
  const b = Math.round(lo[1][2] + t * (hi[1][2] - lo[1][2]))
  const alpha = 0.25 + v * 0.55  // 0.25 at cold end, 0.80 at hot end
  return `rgba(${r},${g},${b},${alpha.toFixed(2)})`
}

const CELL = 32
const LABEL = 20

export default function HeatmapView({ data }: { data: HeatmapData }) {
  const dim = data.board.dimension
  const W = LABEL + dim * CELL + 2
  const H = LABEL + dim * CELL + 2

  return (
    <div className="heatmap-container">
      <div style={{ position: 'relative', display: 'inline-block' }}>
        {/* Base board — desaturated so the heat overlay stands out */}
        <div style={{ filter: 'grayscale(75%) brightness(1.05)' }}>
          <Board data={data.board} />
        </div>

        {/* Heat overlay — absolutely positioned SVG on top */}
        <svg
          width={W}
          height={H}
          style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}
        >
          {data.values.map((row, r) =>
            row.map((v, c) => {
              if (v <= 0) return null
              return (
                <rect
                  key={`${r}-${c}`}
                  x={LABEL + c * CELL}
                  y={LABEL + r * CELL}
                  width={CELL}
                  height={CELL}
                  fill={heatColor(v)}
                />
              )
            })
          )}
        </svg>
      </div>

      <div className="heatmap-legend">
        <span className="heatmap-play">Heatmap: {data.play.trim()}</span>
        <svg width={120} height={16} style={{ display: 'block', marginTop: 4 }}>
          <defs>
            <linearGradient id="heat-grad" x1="0" x2="1" y1="0" y2="0">
              <stop offset="0%"   stopColor="#3b82f6" stopOpacity="0.4" />
              <stop offset="33%"  stopColor="#06b6d4" stopOpacity="0.6" />
              <stop offset="66%"  stopColor="#f59e0b" stopOpacity="0.75" />
              <stop offset="100%" stopColor="#ef4444" stopOpacity="0.85" />
            </linearGradient>
          </defs>
          <rect x={0} y={2} width={120} height={12} rx={3} fill="url(#heat-grad)" />
          <text x={0}   y={28} fontSize={9} fill="#8b949e">cold</text>
          <text x={100} y={28} fontSize={9} fill="#8b949e">hot</text>
        </svg>
      </div>
    </div>
  )
}
