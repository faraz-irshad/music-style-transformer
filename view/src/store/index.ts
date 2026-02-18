import { create } from 'zustand';

interface AudioFile {
  file: File;
  url: string;
  duration: number;
}

interface ExportHistoryItem {
  id: string;
  timestamp: Date;
  styleName: string;
  filename: string;
  url: string;
}

interface AppState {
  audioFile: AudioFile | null;
  selectedStyle: string | null;
  intensity: number;
  pitchShift: number;
  tempoScale: number;
  vibrato: boolean;
  swing: boolean;
  autoMode: boolean;
  isProcessing: boolean;
  progress: number;
  previewUrl: string | null;
  exportHistory: ExportHistoryItem[];
  setAudioFile: (file: AudioFile | null) => void;
  setSelectedStyle: (style: string | null) => void;
  setIntensity: (value: number) => void;
  setPitchShift: (value: number) => void;
  setTempoScale: (value: number) => void;
  setVibrato: (value: boolean) => void;
  setSwing: (value: boolean) => void;
  setAutoMode: (value: boolean) => void;
  setIsProcessing: (value: boolean) => void;
  setProgress: (value: number) => void;
  setPreviewUrl: (url: string | null) => void;
  addExport: (item: ExportHistoryItem) => void;
  reset: () => void;
}

const initialState = {
  audioFile: null,
  selectedStyle: null,
  intensity: 0.7,
  pitchShift: 0,
  tempoScale: 1.0,
  vibrato: false,
  swing: false,
  autoMode: true,
  isProcessing: false,
  progress: 0,
  previewUrl: null,
  exportHistory: [],
};

export const useStore = create<AppState>((set) => ({
  ...initialState,
  setAudioFile: (file) => set({ audioFile: file }),
  setSelectedStyle: (style) => set({ selectedStyle: style }),
  setIntensity: (value) => set({ intensity: value }),
  setPitchShift: (value) => set({ pitchShift: value }),
  setTempoScale: (value) => set({ tempoScale: value }),
  setVibrato: (value) => set({ vibrato: value }),
  setSwing: (value) => set({ swing: value }),
  setAutoMode: (value) => set({ autoMode: value }),
  setIsProcessing: (value) => set({ isProcessing: value }),
  setProgress: (value) => set({ progress: value }),
  setPreviewUrl: (url) => set({ previewUrl: url }),
  addExport: (item) => set((state) => ({
    exportHistory: [item, ...state.exportHistory].slice(0, 5)
  })),
  reset: () => set(initialState),
}));
