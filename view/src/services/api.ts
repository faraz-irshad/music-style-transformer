// Mock API - will be replaced with controller endpoints

interface Style {
  id: string;
  name: string;
  description: string;
  icon: string;
}

const STYLES: Style[] = [
  { id: 'jazz', name: 'Jazz', description: 'Smooth jazz with swing', icon: '🎷' },
  { id: 'classical', name: 'Classical', description: 'Orchestral elegance', icon: '🎻' },
  { id: 'blues', name: 'Blues', description: 'Soulful blues style', icon: '🎸' },
  { id: 'rock', name: 'Rock', description: 'Energetic rock', icon: '🎸' },
  { id: 'electronic', name: 'Electronic', description: 'Modern electronic', icon: '🎹' },
];

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

async function generateMockAudio(duration: number): Promise<string> {
  if (typeof AudioContext === 'undefined') {
    return Promise.resolve(URL.createObjectURL(new Blob([], { type: 'audio/wav' })));
  }
  const audioContext = new AudioContext();
  const sampleRate = audioContext.sampleRate;
  const buffer = audioContext.createBuffer(2, duration * sampleRate, sampleRate);
  
  for (let channel = 0; channel < buffer.numberOfChannels; channel++) {
    const data = buffer.getChannelData(channel);
    for (let i = 0; i < data.length; i++) {
      data[i] = Math.sin(2 * Math.PI * 440 * i / sampleRate) * 0.1;
    }
  }
  
  const blob = await bufferToWav(buffer);
  return URL.createObjectURL(blob);
}

async function bufferToWav(buffer: AudioBuffer): Promise<Blob> {
  const length = buffer.length * buffer.numberOfChannels * 2;
  const arrayBuffer = new ArrayBuffer(44 + length);
  const view = new DataView(arrayBuffer);
  
  const writeString = (offset: number, string: string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };
  
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + length, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, buffer.numberOfChannels, true);
  view.setUint32(24, buffer.sampleRate, true);
  view.setUint32(28, buffer.sampleRate * buffer.numberOfChannels * 2, true);
  view.setUint16(32, buffer.numberOfChannels * 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, length, true);
  
  let offset = 44;
  for (let i = 0; i < buffer.length; i++) {
    for (let channel = 0; channel < buffer.numberOfChannels; channel++) {
      const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
      offset += 2;
    }
  }
  
  return new Blob([arrayBuffer], { type: 'audio/wav' });
}

export const api = {
  async getStyles(): Promise<Style[]> {
    await delay(200);
    return STYLES;
  },

  async transformPreview(params: any, onProgress?: (progress: number) => void): Promise<string> {
    for (let i = 0; i <= 100; i += 10) {
      await delay(200);
      onProgress?.(i);
    }
    return generateMockAudio(5);
  },

  async transformFull(params: any, onProgress?: (progress: number) => void): Promise<string> {
    for (let i = 0; i <= 100; i += 5) {
      await delay(200);
      onProgress?.(i);
    }
    return generateMockAudio(10);
  },
};

export type { Style };
