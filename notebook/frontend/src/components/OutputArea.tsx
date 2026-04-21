import type { NotebookOutput, BoardData, MoveTableData } from '../types'
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

function OutputItem({ output }: { output: NotebookOutput }) {
  switch (output.kind) {
    case 'board':
      return <Board data={output.data as BoardData} />

    case 'table':
      return <MoveTable data={output.data as MoveTableData} />

    case 'text': {
      // Strip ANSI escape codes for clean terminal output display
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

// Strip ANSI escape codes (colors, bold, etc.) from terminal output
function stripAnsi(str: string): string {
  // eslint-disable-next-line no-control-regex
  return str.replace(/\x1b\[[0-9;]*[a-zA-Z]/g, '')
}
