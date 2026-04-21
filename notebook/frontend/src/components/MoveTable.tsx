import { useState } from 'react'
import type { MoveTableData, MoveRow } from '../types'

type SortKey = keyof Pick<MoveRow, 'rank' | 'score' | 'equity'>

export default function MoveTable({ data }: { data: MoveTableData }) {
  const [sortKey, setSortKey] = useState<SortKey>('rank')
  const [sortAsc, setSortAsc] = useState(true)

  const toggle = (key: SortKey) => {
    if (sortKey === key) {
      setSortAsc(a => !a)
    } else {
      setSortKey(key)
      setSortAsc(key === 'rank') // rank asc, others desc by default
    }
  }

  const sorted = [...data.moves].sort((a, b) => {
    const diff = a[sortKey] - b[sortKey]
    return sortAsc ? diff : -diff
  })

  const arrow = (key: SortKey) =>
    sortKey === key ? (sortAsc ? ' ↑' : ' ↓') : ''

  return (
    <table className="move-table">
      <thead>
        <tr>
          <th className="rank" onClick={() => toggle('rank')} style={{ cursor: 'pointer' }}>
            #{arrow('rank')}
          </th>
          <th>Move</th>
          <th>Leave</th>
          <th className="right" onClick={() => toggle('score')} style={{ cursor: 'pointer' }}>
            Score{arrow('score')}
          </th>
          <th className="right" onClick={() => toggle('equity')} style={{ cursor: 'pointer' }}>
            Equity{arrow('equity')}
          </th>
        </tr>
      </thead>
      <tbody>
        {sorted.map(row => (
          <tr key={row.rank}>
            <td className="rank">{row.rank}</td>
            <td className="move-name">{row.move.trim()}</td>
            <td className="leave">{row.leave}</td>
            <td className="right">{row.score}</td>
            <td className={`right ${row.equity >= 0 ? 'equity-positive' : ''}`}>
              {row.equity.toFixed(2)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
