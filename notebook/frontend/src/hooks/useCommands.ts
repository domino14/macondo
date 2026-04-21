import { useState, useEffect } from 'react'

export interface CommandInfo {
  name: string
  description: string
  helpText: string
  options: string[]   // e.g. ["--plies", "--threads"]
  verbs: string[]     // e.g. ["stop", "show"]
}

let cachedCommands: CommandInfo[] | null = null

export function useCommands(): CommandInfo[] {
  const [commands, setCommands] = useState<CommandInfo[]>(cachedCommands ?? [])

  useEffect(() => {
    if (cachedCommands) return
    fetch('/api/commands')
      .then(r => r.json())
      .then((data: CommandInfo[]) => {
        cachedCommands = data
        setCommands(data)
      })
      .catch(() => {})
  }, [])

  return commands
}

// Also export a getter for use outside React (e.g. inside CodeMirror extension)
export function getCommands(): CommandInfo[] {
  return cachedCommands ?? []
}
