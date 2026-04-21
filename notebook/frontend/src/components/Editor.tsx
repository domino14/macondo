import { useEffect, useRef } from 'react'
import { EditorView, keymap, lineNumbers, drawSelection } from '@codemirror/view'
import { EditorState } from '@codemirror/state'
import { defaultKeymap, history, historyKeymap } from '@codemirror/commands'
import { HighlightStyle, syntaxHighlighting, StreamLanguage } from '@codemirror/language'
import { tags as t } from '@lezer/highlight'
import { autocompletion, CompletionContext, CompletionResult } from '@codemirror/autocomplete'
import { getCommands } from '../hooks/useCommands'

// Module-level ref to the most recently focused EditorView.
// Used by CommandPalette to insert text into the active editor.
export let activeEditorView: EditorView | null = null

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

// ── Autocomplete ──────────────────────────────────────────────────────────

const COL_LABELS = 'ABCDEFGHIJKLMNOPQRSTU'

function macondoCompletions(ctx: CompletionContext): CompletionResult | null {
  const line = ctx.state.doc.lineAt(ctx.pos)
  const lineText = line.text.slice(0, ctx.pos - line.from)

  // Skip comments
  if (lineText.trimStart().startsWith('#')) return null

  const tokens = lineText.trimStart().split(/\s+/)
  const isFirstToken = tokens.length === 1
  const currentToken = tokens[tokens.length - 1]

  // Complete command names at start of line
  if (isFirstToken) {
    const commands = getCommands()
    if (commands.length === 0) return null
    const from = line.from + lineText.length - currentToken.length
    return {
      from,
      options: commands.map(c => ({
        label: c.name,
        detail: c.description ? c.description.slice(0, 60) : undefined,
        type: 'keyword',
      })),
      validFor: /^\w[\w-]*$/,
    }
  }

  const cmdName = tokens[0]
  const commands = getCommands()
  const spec = commands.find(c => c.name === cmdName)

  // Complete --flags for known commands
  if (currentToken.startsWith('-') && spec) {
    const from = ctx.pos - currentToken.length
    const options = [
      ...spec.options.map(o => ({ label: o, type: 'property' as const })),
      ...spec.verbs.map(v => ({ label: v, type: 'keyword' as const })),
    ]
    return { from, options, validFor: /^-[\w-]*$/ }
  }

  // Complete board coordinates (e.g. 8D or D8) after certain commands
  const coordCmds = new Set(['turn', 'commit', 'add'])
  if (coordCmds.has(cmdName) && /^\d*$/.test(currentToken)) {
    const from = ctx.pos - currentToken.length
    const opts = []
    for (let r = 1; r <= 15; r++) {
      for (const c of COL_LABELS.slice(0, 15)) {
        opts.push({ label: `${r}${c}`, type: 'variable' as const })
      }
    }
    return { from, options: opts, validFor: /^\d*[A-Oa-o]?$/ }
  }

  return null
}

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
        autocompletion({ override: [macondoCompletions], activateOnTyping: true }),
        EditorView.updateListener.of(update => {
          if (update.docChanged) {
            onChangeRef.current(update.state.doc.toString())
          }
        }),
        EditorView.domEventHandlers({
          focus: () => { activeEditorView = view },
          // No blur clear — activeEditorView persists as "last focused editor"
          // so the command palette can still insert after Cmd+P steals focus.
        }),
        EditorView.lineWrapping,
      ],
    })

    const view = new EditorView({ state, parent: containerRef.current })
    viewRef.current = view

    if (autoFocus) {
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
