import { useState, useEffect, useRef } from 'react'
import type { CellData, CellLanguage, NotebookOutput } from '../types'
import Editor from './Editor'
import OutputArea from './OutputArea'

interface ProgressData { message: string; iterations?: number }

interface CellProps {
  cell: CellData
  onContentChange: (id: string, content: string) => void
  onLanguageChange: (id: string, lang: CellLanguage) => void
  onExecute: (id: string) => void
  onCancel: (id: string) => void
  onDelete: (id: string) => void
  onAddBelow: (id: string) => void
  // Outputs are managed externally (App-level SSE routing)
  outputs: NotebookOutput[]
  isRunning: boolean
  progress?: ProgressData
  autoFocus?: boolean
  onFocused?: () => void
}

export default function Cell({
  cell, onContentChange, onLanguageChange, onExecute, onCancel, onDelete, onAddBelow,
  outputs, isRunning, progress, autoFocus, onFocused,
}: CellProps) {
  const [isMarkdownEditing, setIsMarkdownEditing] = useState(false)
  const markdownEditorRef = useRef<HTMLDivElement>(null)

  // Close markdown editor when clicking outside it
  useEffect(() => {
    if (!isMarkdownEditing) return
    const handleMouseDown = (e: MouseEvent) => {
      if (markdownEditorRef.current && !markdownEditorRef.current.contains(e.target as Node)) {
        setIsMarkdownEditing(false)
      }
    }
    document.addEventListener('mousedown', handleMouseDown)
    return () => document.removeEventListener('mousedown', handleMouseDown)
  }, [isMarkdownEditing])

  const toggleLang = () => {
    onLanguageChange(cell.id, cell.language === 'macondo' ? 'markdown' : 'macondo')
  }

  const run = () => onExecute(cell.id)

  return (
    <div className={`cell${isRunning ? ' running' : ''}`}>
      {/* ── Toolbar ── */}
      <div className="cell-toolbar">
        <span
          className={`cell-lang-badge${cell.language === 'markdown' ? ' markdown' : ''}`}
          onClick={toggleLang}
          title="Click to toggle cell type"
        >
          {cell.language}
        </span>

        {isRunning && progress && (
          <span className="cell-progress-msg">{progress.message}</span>
        )}

        <span className="cell-toolbar-spacer" />

        {isRunning ? (
          <button
            className="cell-btn cancel"
            onClick={() => onCancel(cell.id)}
            title="Cancel execution"
          >
            ■ Cancel
          </button>
        ) : (
          <button
            className="cell-btn run"
            onClick={run}
            title="Run cell (Shift+Enter)"
          >
            ▶ Run
          </button>
        )}

        <button
          className="cell-btn"
          onClick={() => onAddBelow(cell.id)}
          title="Add cell below"
        >
          +
        </button>

        <button
          className="cell-btn"
          onClick={() => onDelete(cell.id)}
          title="Delete cell"
        >
          ✕
        </button>
      </div>

      {/* ── Input area ── */}
      {cell.language === 'macondo' ? (
        <Editor
          value={cell.content}
          onChange={v => onContentChange(cell.id, v)}
          onExecute={run}
          autoFocus={autoFocus}
          onFocus={onFocused}
        />
      ) : (
        isMarkdownEditing ? (
          <div ref={markdownEditorRef}>
            <Editor
              value={cell.content}
              onChange={v => onContentChange(cell.id, v)}
              onExecute={() => { setIsMarkdownEditing(false); run() }}
            />
          </div>
        ) : (
          <div
            className="cell-markdown"
            onClick={() => setIsMarkdownEditing(true)}
            dangerouslySetInnerHTML={{ __html: renderMarkdown(cell.content || '*Click to edit*') }}
          />
        )
      )}

      {/* ── Outputs ── */}
      <OutputArea outputs={outputs} />
    </div>
  )
}

// Very simple markdown renderer (no external dep needed for Phase 2)
function renderMarkdown(md: string): string {
  return md
    // Code blocks
    .replace(/```[\s\S]*?```/g, m => `<pre><code>${escHtml(m.slice(3, -3).replace(/^[a-z]*\n/, ''))}</code></pre>`)
    // Inline code
    .replace(/`([^`]+)`/g, (_, c) => `<code>${escHtml(c)}</code>`)
    // Headers
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    // Bold / italic
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    // Paragraphs (double newline)
    .replace(/\n\n+/g, '</p><p>')
    .replace(/^(?!<[hp])/, '<p>')
    .concat('</p>')
}

function escHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}
