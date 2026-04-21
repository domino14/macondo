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

export interface TileInfo {
  letter: string;
  playthrough: boolean;
}

export interface MoveRow {
  rank: number;
  move: string;
  leave: string;
  score: number;
  equity: number;
  rowStart: number;
  colStart: number;
  isVert: boolean;
  tiles?: TileInfo[];
}

export interface MoveTableData {
  moves: MoveRow[];
  board?: BoardData;
}

export interface HeatmapData {
  board: BoardData;
  values: number[][];  // dim×dim, 0..1
  play: string;
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
