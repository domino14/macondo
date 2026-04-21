import { useState, useCallback, useEffect, useRef, useId } from 'react'
import type { CellData, CellLanguage, NotebookOutput, SSEEvent } from './types'
import { useSSE } from './hooks/useSSE'
import Cell from './components/Cell'

interface ProgressData { message: string; iterations?: number }

function makeCell(content = '', language: CellLanguage = 'macondo'): CellData {
  return {
    id: crypto.randomUUID(),
    content,
    language,
    outputs: [],
    isRunning: false,
  }
}

const INITIAL_CELLS: CellData[] = [
  makeCell('# Load a game and explore\n# Shift+Enter or ▶ to run', 'markdown'),
  makeCell('new\ngen 15'),
]

export default function App() {
  const [cells, setCells] = useState<CellData[]>(INITIAL_CELLS)
  const [focusCellId, setFocusCellId] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const prevCellCountRef = useRef(INITIAL_CELLS.length)

  useEffect(() => {
    if (cells.length > prevCellCountRef.current) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
    prevCellCountRef.current = cells.length
  }, [cells.length])

  // App-level output/running/progress state, keyed by cell id
  const [cellOutputs, setCellOutputs] = useState<Record<string, NotebookOutput[]>>({})
  const [runningCells, setRunningCells] = useState<Set<string>>(new Set())
  const [cellProgress, setCellProgress] = useState<Record<string, ProgressData>>({})

  // ── SSE event routing ──────────────────────────────────────────────────
  const handleSSEEvent = useCallback((event: SSEEvent) => {
    const { cell_id } = event

    if (event.output) {
      if (event.output.kind === 'progress') {
        // Update progress in place — don't accumulate in the output list
        setCellProgress(prev => ({ ...prev, [cell_id]: event.output!.data as ProgressData }))
      } else {
        setCellOutputs(prev => ({
          ...prev,
          [cell_id]: [...(prev[cell_id] ?? []), event.output!],
        }))
      }
    }

    if (event.error) {
      setCellOutputs(prev => ({
        ...prev,
        [cell_id]: [
          ...(prev[cell_id] ?? []),
          { kind: 'error', data: event.error! },
        ],
      }))
    }

    if (event.done) {
      setRunningCells(prev => {
        const next = new Set(prev)
        next.delete(cell_id)
        return next
      })
      setCellProgress(prev => {
        const next = { ...prev }
        delete next[cell_id]
        return next
      })
    }
  }, [])

  useSSE(handleSSEEvent)

  // ── Cell operations ────────────────────────────────────────────────────
  const handleContentChange = useCallback((id: string, content: string) => {
    setCells(prev => prev.map(c => c.id === id ? { ...c, content } : c))
  }, [])

  const handleLanguageChange = useCallback((id: string, lang: CellLanguage) => {
    setCells(prev => prev.map(c => c.id === id ? { ...c, language: lang } : c))
  }, [])

  const handleExecute = useCallback(async (id: string) => {
    const cell = cells.find(c => c.id === id)
    if (!cell || runningCells.has(id)) return

    // Clear old outputs/progress, mark running
    setCellOutputs(prev => ({ ...prev, [id]: [] }))
    setCellProgress(prev => { const n = { ...prev }; delete n[id]; return n })
    setRunningCells(prev => new Set([...prev, id]))

    try {
      await fetch('/api/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          cell_id: id,
          content: cell.content,
          language: cell.language,
        }),
      })
    } catch (err) {
      setCellOutputs(prev => ({
        ...prev,
        [id]: [{ kind: 'error', data: String(err) }],
      }))
      setRunningCells(prev => {
        const next = new Set(prev)
        next.delete(id)
        return next
      })
    }
  }, [cells, runningCells])

  const handleRunAll = useCallback(async () => {
    for (const cell of cells) {
      await handleExecute(cell.id)
      // Wait for this cell to finish before running the next
      await waitForDone(cell.id, runningCells)
    }
  }, [cells, handleExecute, runningCells])

  const handleDelete = useCallback((id: string) => {
    setCells(prev => {
      if (prev.length <= 1) return prev // always keep at least one cell
      return prev.filter(c => c.id !== id)
    })
  }, [])

  const handleAddBelow = useCallback((id: string) => {
    const newCell = makeCell()
    setCells(prev => {
      const idx = prev.findIndex(c => c.id === id)
      const next = [...prev]
      next.splice(idx + 1, 0, newCell)
      return next
    })
    setFocusCellId(newCell.id)
  }, [])

  const handleCancel = useCallback(async (id: string) => {
    try {
      await fetch('/api/cancel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cell_id: id }),
      })
    } catch {
      // ignore
    }
  }, [])

  const handleAddCell = () => {
    const newCell = makeCell()
    setCells(prev => [...prev, newCell])
    setFocusCellId(newCell.id)
  }

  // ── Notebook save/load ─────────────────────────────────────────────────
  const handleSave = useCallback(() => {
    const nb = {
      version: 1,
      cells: cells.map(c => ({ id: c.id, content: c.content, language: c.language })),
      outputs: cellOutputs,
    }
    const blob = new Blob([JSON.stringify(nb, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'notebook.macondo-nb.json'
    a.click()
    URL.revokeObjectURL(url)
  }, [cells, cellOutputs])

  const fileInputId = useId()
  const handleLoad = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = ev => {
      try {
        const nb = JSON.parse(ev.target!.result as string)
        const loadedCells: CellData[] = (nb.cells ?? []).map((c: CellData) => ({
          ...makeCell(c.content, c.language),
          id: c.id,
        }))
        setCells(loadedCells.length > 0 ? loadedCells : [makeCell()])
        setCellOutputs(nb.outputs ?? {})
        setCellProgress({})
        setRunningCells(new Set())
      } catch {
        alert('Could not parse notebook file.')
      }
    }
    reader.readAsText(file)
    e.target.value = '' // reset so the same file can be re-loaded
  }, [])

  return (
    <>
      <header className="header">
        <div className="header-brand">
          <div className="header-dot" />
          Macondo Notebook
        </div>
        <span className="header-spacer" />
        <label htmlFor={fileInputId} className="header-btn" style={{ cursor: 'pointer' }}>
          Open
          <input
            id={fileInputId}
            type="file"
            accept=".json,.macondo-nb.json"
            style={{ display: 'none' }}
            onChange={handleLoad}
          />
        </label>
        <button className="header-btn" onClick={handleSave}>Save</button>
        <button className="header-btn" onClick={handleAddCell}>+ Cell</button>
        <button className="header-btn primary" onClick={handleRunAll}>▶▶ Run All</button>
      </header>

      <main className="notebook">
        {cells.map(cell => (
          <Cell
            key={cell.id}
            cell={cell}
            onContentChange={handleContentChange}
            onLanguageChange={handleLanguageChange}
            onExecute={handleExecute}
            onCancel={handleCancel}
            onDelete={handleDelete}
            onAddBelow={handleAddBelow}
            outputs={cellOutputs[cell.id] ?? []}
            isRunning={runningCells.has(cell.id)}
            progress={cellProgress[cell.id]}
            autoFocus={focusCellId === cell.id}
            onFocused={() => setFocusCellId(null)}
          />
        ))}

        <button className="add-cell" onClick={handleAddCell}>
          + Add Cell
        </button>
        <div ref={bottomRef} />
      </main>
    </>
  )
}

// Poll until a cell is no longer running (used by Run All)
function waitForDone(id: string, running: Set<string>): Promise<void> {
  if (!running.has(id)) return Promise.resolve()
  return new Promise(resolve => {
    const check = () => {
      if (!running.has(id)) resolve()
      else setTimeout(check, 100)
    }
    setTimeout(check, 100)
  })
}
