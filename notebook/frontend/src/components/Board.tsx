import type { BoardData } from '../types'

// Bonus square colors — matched exactly to the 3D renderer defaults (render_template.html).
// Board background is jade (#00ffbd), tile color is orange (#ff6b35).
const BONUS: Record<string, { fill: string; label: string; text: string }> = {
  '=': { fill: '#cc5555', label: '3W', text: '#fff' },   // Triple word  — desaturated red
  '-': { fill: '#ff9999', label: '2W', text: '#7a1a1a' }, // Double word  — light red
  '"': { fill: '#5566cc', label: '3L', text: '#fff' },   // Triple letter — desaturated blue
  "'": { fill: '#4eb7e1', label: '2L', text: '#0a2a38' }, // Double letter — sky blue
  '~': { fill: '#22ff22', label: '4W', text: '#003300' }, // Quad word
  '^': { fill: '#99ff99', label: '4L', text: '#003300' }, // Quad letter
  '*': { fill: '#00ffbd', label: '',   text: '' },        // Starting square (same as board)
  ' ': { fill: '#00ffbd', label: '',   text: '' },        // No bonus — jade board
}

const CELL = 32          // px per cell
const LABEL = 20         // px for row/col labels
const COL_LABELS = 'ABCDEFGHIJKLMNOPQRSTU'
const TILE_RADIUS = 2

/** Parse FEN string into a 2-D array of cell content ('', 'A', 'a', '[CH]', …). */
function parseFEN(fen: string): string[][] {
  return fen.split('/').map(row => {
    const cells: string[] = []
    let i = 0
    while (i < row.length) {
      if (row[i] === '[') {
        let tile = ''
        i++
        while (i < row.length && row[i] !== ']') tile += row[i++]
        i++
        cells.push(tile)
      } else if (row[i] >= '0' && row[i] <= '9') {
        let num = ''
        while (i < row.length && row[i] >= '0' && row[i] <= '9') num += row[i++]
        cells.push(...Array(parseInt(num)).fill(''))
      } else {
        cells.push(row[i++])
      }
    }
    return cells
  })
}

// Tile colors match the 3D renderer: orange (#ff6b35) tiles, black text.
// Blank tiles use a lighter orange with blue text (matching 3D renderer's 0x0000ff for blanks).
const TILE_BG  = '#ff6b35'
const TILE_FG  = '#000000'

function TileSVG({
  x, y, letter, score, isBlank,
}: { x: number; y: number; letter: string; score: number; isBlank: boolean }) {
  const bg = TILE_BG
  const fg = TILE_FG
  const pad = 1.5
  return (
    <g>
      <rect
        x={x + pad} y={y + pad}
        width={CELL - pad * 2} height={CELL - pad * 2}
        fill={bg}
        rx={TILE_RADIUS}
        stroke="rgba(0,0,0,0.25)"
        strokeWidth={0.5}
      />
      {/* Bottom edge shadow for depth */}
      <rect
        x={x + pad} y={y + CELL - pad - 2}
        width={CELL - pad * 2} height={2}
        fill="rgba(0,0,0,0.15)"
        rx={TILE_RADIUS}
      />
      <text
        x={x + CELL / 2}
        y={y + CELL / 2 + 1}
        textAnchor="middle"
        dominantBaseline="middle"
        fontSize={isBlank ? 13 : 14}
        fontWeight="600"
        fontFamily="'Inter', sans-serif"
        fill={fg}
      >
        {letter.toUpperCase()}
      </text>
      {!isBlank && score > 0 && (
        <text
          x={x + CELL - 4}
          y={y + CELL - 4}
          textAnchor="end"
          fontSize={8}
          fontFamily="'Inter', sans-serif"
          fill={fg}
          opacity={0.6}
        >
          {score}
        </text>
      )}
    </g>
  )
}

function RackTile({ letter, score, x, y }: { letter: string; score: number; x: number; y: number }) {
  const isBlank = letter === '?'
  return (
    <g>
      <rect x={x} y={y} width={CELL + 2} height={CELL + 2}
        fill={TILE_BG} rx={TILE_RADIUS + 1}
        stroke="rgba(0,0,0,0.25)" strokeWidth={1}
      />
      <rect x={x} y={y + CELL - 1} width={CELL + 2} height={3}
        fill="rgba(0,0,0,0.15)" rx={2}
      />
      <text
        x={x + (CELL + 2) / 2} y={y + (CELL + 2) / 2 + 1}
        textAnchor="middle" dominantBaseline="middle"
        fontSize={14} fontWeight="600"
        fontFamily="'Inter', sans-serif"
        fill={TILE_FG}
      >
        {isBlank ? '?' : letter}
      </text>
      {score > 0 && (
        <text
          x={x + CELL - 2} y={y + CELL - 2}
          textAnchor="end" fontSize={8}
          fontFamily="'Inter', sans-serif"
          fill={TILE_FG} opacity={0.6}
        >
          {score}
        </text>
      )}
    </g>
  )
}

export default function Board({ data }: { data: BoardData }) {
  const dim = data.dimension
  const board = parseFEN(data.fen)
  const W = LABEL + dim * CELL + 2
  const H = LABEL + dim * CELL + 2

  // Rack tiles
  const rackLetters = data.rack ? data.rack.split('') : []
  const rackW = rackLetters.length * (CELL + 6)

  return (
    <div className="board-container">
      <div className="board-svg-wrap">
        {/* Main board */}
        <svg
          className="board-svg"
          width={W}
          height={H}
          viewBox={`0 0 ${W} ${H}`}
          style={{ borderRadius: 4 }}
        >
          {/* Board background */}
          <rect x={LABEL} y={LABEL} width={dim * CELL} height={dim * CELL}
            fill="var(--board-bg)" />

          {/* Column labels */}
          {Array.from({ length: dim }, (_, c) => (
            <text
              key={c}
              x={LABEL + c * CELL + CELL / 2}
              y={LABEL - 5}
              textAnchor="middle"
              fontSize={10}
              fontFamily="'Inter', sans-serif"
              fill="var(--text-secondary)"
            >
              {COL_LABELS[c]}
            </text>
          ))}

          {/* Row labels */}
          {Array.from({ length: dim }, (_, r) => (
            <text
              key={r}
              x={LABEL - 4}
              y={LABEL + r * CELL + CELL / 2 + 1}
              textAnchor="end"
              dominantBaseline="middle"
              fontSize={10}
              fontFamily="'Inter', sans-serif"
              fill="var(--text-secondary)"
            >
              {r + 1}
            </text>
          ))}

          {/* Cells */}
          {Array.from({ length: dim }, (_, row) =>
            Array.from({ length: dim }, (_, col) => {
              const x = LABEL + col * CELL
              const y = LABEL + row * CELL
              const bonusChar = data.bonusLayout[row]?.[col] ?? ' '
              const bonus = BONUS[bonusChar] ?? BONUS[' ']
              const cell = board[row]?.[col] ?? ''
              const hasTile = cell !== ''
              const isBlank = hasTile && cell === cell.toLowerCase() && cell !== cell.toUpperCase()
              const displayLetter = cell ? cell.replace(/[\[\]]/g, '') : ''
              const score = cell ? (data.alphabetScores[displayLetter.toUpperCase()] ?? 0) : 0

              return (
                <g key={`${row}-${col}`}>
                  {/* Bonus square background */}
                  <rect x={x} y={y} width={CELL} height={CELL}
                    fill={bonus.fill}
                    stroke="rgba(0,0,0,0.08)"
                    strokeWidth={0.5}
                  />
                  {/* Bonus label (only when empty) */}
                  {!hasTile && bonus.label && (
                    <text
                      x={x + CELL / 2} y={y + CELL / 2 + 1}
                      textAnchor="middle" dominantBaseline="middle"
                      fontSize={8} fontWeight="600"
                      fontFamily="'Inter', sans-serif"
                      fill={bonus.text}
                      opacity={0.85}
                    >
                      {bonus.label}
                    </text>
                  )}
                  {/* Tile */}
                  {hasTile && (
                    <TileSVG
                      x={x} y={y}
                      letter={displayLetter}
                      score={score}
                      isBlank={isBlank}
                    />
                  )}
                </g>
              )
            })
          )}

          {/* Grid lines (subtle) */}
          {Array.from({ length: dim + 1 }, (_, i) => (
            <line key={`v${i}`}
              x1={LABEL + i * CELL} y1={LABEL}
              x2={LABEL + i * CELL} y2={LABEL + dim * CELL}
              stroke="rgba(0,0,0,0.12)" strokeWidth={0.5}
            />
          ))}
          {Array.from({ length: dim + 1 }, (_, i) => (
            <line key={`h${i}`}
              x1={LABEL} y1={LABEL + i * CELL}
              x2={LABEL + dim * CELL} y2={LABEL + i * CELL}
              stroke="rgba(0,0,0,0.12)" strokeWidth={0.5}
            />
          ))}
        </svg>

        {/* Rack */}
        {rackLetters.length > 0 && (
          <svg
            width={rackW + LABEL}
            height={CELL + 12}
            style={{ marginTop: 8, display: 'block' }}
          >
            <text x={LABEL - 4} y={(CELL + 12) / 2 + 1}
              textAnchor="end" dominantBaseline="middle"
              fontSize={10} fontFamily="'Inter', sans-serif"
              fill="var(--text-secondary)"
            >
              ♟
            </text>
            {rackLetters.map((letter, i) => (
              <RackTile
                key={i}
                letter={letter}
                score={data.alphabetScores[letter.toUpperCase()] ?? 0}
                x={LABEL + i * (CELL + 6)}
                y={4}
              />
            ))}
          </svg>
        )}
      </div>

      {/* Side info panel */}
      <div className="board-info">
        {data.players.map((p, i) => (
          <div
            key={i}
            className={`board-player${i === data.playerOnTurn ? ' on-turn' : ''}`}
          >
            <span className="board-player-name">
              {i === data.playerOnTurn && <span className="board-player-arrow">▶ </span>}
              {p.name}
            </span>
            <span className="board-player-score">{p.score}</span>
          </div>
        ))}

        <div className="board-meta">
          <span>Turn <b>{data.turnNumber}</b></span>
          <span>Bag <b>{data.bagCount}</b></span>
        </div>

        {data.lastPlay && (
          <div className="board-last-play">{data.lastPlay}</div>
        )}
      </div>
    </div>
  )
}
