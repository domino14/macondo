import { useState } from 'react'
import type { NotebookOutput, BoardData, MoveTableData, MoveRow } from '../types'
import Board from './Board'
import MoveTable from './MoveTable'

export default function OutputArea({ outputs }: { outputs: NotebookOutput[] }) {
  if (outputs.length === 0) return null

  return (
    <div className="cell-outputs">
      {outputs.map((out, i) => (
        <div key={i} className="output-item">
          <OutputItem output={out} />
        </div>
      ))}
    </div>
  )
}

// MoveExplorer renders a move table and board side-by-side.
// Clicking a row highlights the move on the board.
function MoveExplorer({ table, board }: { table: MoveTableData; board: BoardData }) {
  const [selected, setSelected] = useState<MoveRow | null>(null)

  return (
    <div className="move-explorer">
      <div className="move-explorer-table">
        <MoveTable
          data={table}
          selectedRank={selected?.rank}
          onSelect={setSelected}
        />
      </div>
      <div className="move-explorer-board">
        <Board data={board} highlightMove={selected} />
      </div>
    </div>
  )
}

function OutputItem({ output }: { output: NotebookOutput }) {
  switch (output.kind) {
    case 'board':
      return <Board data={output.data as BoardData} />

    case 'table': {
      const td = output.data as MoveTableData
      return td.board
        ? <MoveExplorer table={td} board={td.board} />
        : <MoveTable data={td} />
    }

    case 'text': {
      const text = stripAnsi(String(output.data))
      return <pre className="output-text">{text}</pre>
    }

    case 'error':
      return <pre className="output-error">Error: {String(output.data)}</pre>

    default:
      return (
        <pre className="output-text">
          {JSON.stringify(output.data, null, 2)}
        </pre>
      )
  }
}

function stripAnsi(str: string): string {
  // eslint-disable-next-line no-control-regex
  return str.replace(/\x1b\[[0-9;]*[a-zA-Z]/g, '')
}
