import { useEffect, useRef } from 'react'
import type { SSEEvent } from '../types'

export function useSSE(onEvent: (e: SSEEvent) => void) {
  // Keep the callback stable via ref so we don't reconnect on every render.
  const callbackRef = useRef(onEvent)
  callbackRef.current = onEvent

  useEffect(() => {
    const es = new EventSource('/api/stream')

    es.onmessage = (e: MessageEvent) => {
      try {
        const event = JSON.parse(e.data) as SSEEvent
        callbackRef.current(event)
      } catch {
        // ignore malformed events
      }
    }

    es.onerror = () => {
      // EventSource auto-reconnects; nothing to do here
    }

    return () => es.close()
  }, []) // connect once
}
