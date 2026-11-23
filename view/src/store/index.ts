import { create } from 'zustand';

export interface Metrics {
  clf_prob_baroque?: number;
  clf_prob_romantic?: number;
  skyline_dom?: number;
  clash_rate?: number;
  pc_entropy?: number;
  fourgram_rep?: number;
}

export interface Output {
  file: string;
  layer: string;
  layer_label: string;
  style: string;
  audio_url: string | null;
  metrics: Metrics;
}

export interface Melody {
  melody_id: string;
  melody_audio: string | null;
  outputs: Output[];
}

export interface LayerInfo {
  id: string;
  label: string;
}

export interface DemoIndex {
  melodies: Melody[];
  summary: Record<string, unknown>;
  layers: LayerInfo[];
  _empty?: boolean;
}

interface AppState {
  index: DemoIndex | null;
  loading: boolean;
  error: string | null;
  selectedMelodyId: string | null;
  abStyle: 'baroque' | 'romantic';
  setIndex: (index: DemoIndex) => void;
  setLoading: (v: boolean) => void;
  setError: (e: string | null) => void;
  setSelectedMelodyId: (id: string | null) => void;
  setAbStyle: (s: 'baroque' | 'romantic') => void;
}

export const useStore = create<AppState>((set) => ({
  index: null,
  loading: true,
  error: null,
  selectedMelodyId: null,
  abStyle: 'baroque',
  setIndex: (index) => set({ index }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
  setSelectedMelodyId: (id) => set({ selectedMelodyId: id }),
  setAbStyle: (abStyle) => set({ abStyle }),
}));
