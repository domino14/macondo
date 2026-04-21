import { useState, useEffect, useRef, useCallback } from 'react'
import type { CommandInfo } from '../hooks/useCommands'
import { activeEditorView } from './Editor'

interface Props {
  commands: CommandInfo[]
  onClose: () => void
}

export default function CommandPalette({ commands, onClose }: Props) {
  const [query, setQuery] = useState('')
  const [selected, setSelected] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLUListElement>(null)

  const filtered = query.trim()
    ? commands.filter(c =>
        c.name.includes(query.trim()) ||
        c.description.toLowerCase().includes(query.trim().toLowerCase())
      )
    : commands

  const current = filtered[selected] ?? null

  useEffect(() => { setSelected(0) }, [query])
  useEffect(() => { inputRef.current?.focus() }, [])
  useEffect(() => {
    const el = listRef.current?.children[selected] as HTMLElement | undefined
    el?.scrollIntoView({ block: 'nearest' })
  }, [selected])

  const insert = useCallback((cmd: CommandInfo) => {
    const view = activeEditorView
    if (view) {
      const doc = view.state.doc
      const pos = doc.length
      const prefix = doc.length > 0 && !doc.toString().endsWith('\n') ? '\n' : ''
      view.dispatch({
        changes: { from: pos, insert: prefix + cmd.name },
        selection: { anchor: pos + prefix.length + cmd.name.length },
      })
      view.focus()
    }
    onClose()
  }, [onClose])

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') { onClose(); return }
    if (e.key === 'ArrowDown') { e.preventDefault(); setSelected(s => Math.min(s + 1, filtered.length - 1)) }
    if (e.key === 'ArrowUp') { e.preventDefault(); setSelected(s => Math.max(s - 1, 0)) }
    if (e.key === 'Enter' && filtered[selected]) { insert(filtered[selected]) }
  }

  return (
    <div className="palette-backdrop" onClick={onClose}>
      <div className="palette-modal" onClick={e => e.stopPropagation()}>
        <input
          ref={inputRef}
          className="palette-input"
          placeholder="Search commands..."
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={handleKey}
        />

        <div className="palette-body">
          {/* Left: command list */}
          <ul ref={listRef} className="palette-list">
            {filtered.length === 0 && (
              <li className="palette-empty">No commands match</li>
            )}
            {filtered.map((cmd, i) => (
              <li
                key={cmd.name}
                className={`palette-item${i === selected ? ' selected' : ''}`}
                // mousedown fires before blur, so activeEditorView is still set;
                // preventDefault keeps focus on the editor (no blur at all).
                onMouseDown={e => { e.preventDefault(); insert(cmd) }}
                onMouseEnter={() => setSelected(i)}
              >
                <span className="palette-cmd">{cmd.name}</span>
                {cmd.description && (
                  <span className="palette-desc">{cmd.description}</span>
                )}
              </li>
            ))}
          </ul>

          {/* Right: help text for selected command */}
          {current?.helpText && (
            <div className="palette-help">
              <div className="palette-help-name">{current.name}</div>
              <pre className="palette-help-text">{current.helpText}</pre>
            </div>
          )}
        </div>

        <div className="palette-hint">
          ↑↓ navigate · Enter / click to insert · Esc close
        </div>
      </div>
    </div>
  )
}
