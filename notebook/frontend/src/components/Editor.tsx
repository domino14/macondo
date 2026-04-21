import { useEffect, useRef } from 'react'
import { EditorView, keymap, lineNumbers, drawSelection } from '@codemirror/view'
import { EditorState } from '@codemirror/state'
import { defaultKeymap, history, historyKeymap } from '@codemirror/commands'
import { HighlightStyle, syntaxHighlighting, StreamLanguage } from '@codemirror/language'
import { tags as t } from '@lezer/highlight'

// Board coordinate pattern: [0-9]+[A-Oa-o] or [A-Oa-o][0-9]+
const coordRe = /^([A-Oa-o]\d+|\d+[A-Oa-o])\b/

const macondoLanguage = StreamLanguage.define({
  name: 'macondo',
  token(stream, state: { atLineStart: boolean }) {
    // Comments
    if (stream.match(/^#.*/)) return 'comment'

    // Skip whitespace
    if (stream.eatSpace()) return null

    // First token on a new line → command keyword
    if (state.atLineStart) {
      state.atLineStart = false
      if (stream.match(/^\S+/)) return 'keyword'
      return null
    }

    // Flags: -something or --something
    if (stream.match(/^--?\w[\w-]*/)) return 'meta'

    // Board coordinates
    if (stream.match(coordRe)) return 'string'

    // Quoted strings
    if (stream.match(/^"[^"]*"/)) return 'string'
    if (stream.match(/^'[^']*'/)) return 'string'

    // Numbers (including negative)
    if (stream.match(/^-?\d+(\.\d+)?/)) return 'number'

    // Other tokens
    stream.match(/^\S+/)
    return null
  },
  startState: () => ({ atLineStart: true }),
  blankLine: (state: { atLineStart: boolean }) => { state.atLineStart = true },
  indent: () => null,
  languageData: {},
})

// ── Theme & highlight styles ───────────────────────────────────────────────

const macondoHighlight = HighlightStyle.define([
  { tag: t.keyword,  class: 'cm-mac-command' },
  { tag: t.meta,     class: 'cm-mac-flag' },
  { tag: t.string,   class: 'cm-mac-coord' },
  { tag: t.number,   class: 'cm-mac-number' },
  { tag: t.comment,  class: 'cm-mac-comment' },
])

const editorTheme = EditorView.theme({
  '&': { height: 'auto' },
  '.cm-scroller': { overflow: 'auto', maxHeight: '300px' },
})

// ── Component ─────────────────────────────────────────────────────────────

interface EditorProps {
  value: string
  onChange: (value: string) => void
  onExecute: () => void
  autoFocus?: boolean
  onFocus?: () => void
}

export default function Editor({ value, onChange, onExecute, autoFocus, onFocus }: EditorProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const viewRef = useRef<EditorView | null>(null)

  // Stable refs to callbacks
  const onChangeRef = useRef(onChange)
  const onExecuteRef = useRef(onExecute)
  onChangeRef.current = onChange
  onExecuteRef.current = onExecute

  useEffect(() => {
    if (!containerRef.current) return

    const executeKeymap = keymap.of([
      {
        key: 'Shift-Enter',
        run: () => { onExecuteRef.current(); return true },
      },
      {
        key: 'Ctrl-Enter',
        run: () => { onExecuteRef.current(); return true },
      },
      {
        key: 'Mod-Enter',
        run: () => { onExecuteRef.current(); return true },
      },
    ])

    const state = EditorState.create({
      doc: value,
      extensions: [
        history(),
        lineNumbers(),
        drawSelection(),
        macondoLanguage,
        syntaxHighlighting(macondoHighlight),
        editorTheme,
        executeKeymap,
        keymap.of([...defaultKeymap, ...historyKeymap]),
        EditorView.updateListener.of(update => {
          if (update.docChanged) {
            onChangeRef.current(update.state.doc.toString())
          }
        }),
        EditorView.lineWrapping,
      ],
    })

    const view = new EditorView({ state, parent: containerRef.current })
    viewRef.current = view

    if (autoFocus) {
      // Defer so the DOM is painted before we steal focus
      requestAnimationFrame(() => {
        view.focus()
        onFocus?.()
      })
    }

    return () => {
      view.destroy()
      viewRef.current = null
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // create once

  // Sync external value changes (e.g., loading a saved notebook)
  useEffect(() => {
    const view = viewRef.current
    if (!view) return
    const current = view.state.doc.toString()
    if (current !== value) {
      view.dispatch({
        changes: { from: 0, to: current.length, insert: value },
      })
    }
  }, [value])

  return <div className="cell-editor" ref={containerRef} />
}
