export type CellLanguage = 'macondo' | 'markdown';

export interface PlayerInfo {
  name: string;
  score: number;
}

export interface BoardData {
  fen: string;
  rack: string;
  playerOnTurn: number;
  players: PlayerInfo[];
  bagCount: number;
  turnNumber: number;
  lastPlay?: string;
  bonusLayout: string[];
  dimension: number;
  alphabetScores: Record<string, number>;
  remainingTiles: Record<string, number>;
}

export interface MoveRow {
  rank: number;
  move: string;
  leave: string;
  score: number;
  equity: number;
}

export interface MoveTableData {
  moves: MoveRow[];
}

export interface NotebookOutput {
  kind: 'board' | 'table' | 'text' | 'error' | 'progress' | string;
  data: BoardData | MoveTableData | string | unknown;
}

export interface SSEEvent {
  cell_id: string;
  output?: NotebookOutput;
  done?: boolean;
  error?: string;
}

export interface CellData {
  id: string;
  content: string;
  language: CellLanguage;
  outputs: NotebookOutput[];
  isRunning: boolean;
}
