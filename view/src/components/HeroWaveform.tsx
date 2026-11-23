import { Stack, Text, Box, Group, Badge } from '@mantine/core';
import { IconWaveSquare, IconMusic } from '@tabler/icons-react';
import { useStore } from '../store';
import { useEffect, useRef } from 'react';

export function HeroWaveform() {
  const { audioFile, selectedStyle } = useStore();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioContextRef = useRef<AudioContext | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !audioFile) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, width, h);

    const drawWaveform = async () => {
      try {
        const audioContext = new AudioContext();
        audioContextRef.current = audioContext;
        const response = await fetch(audioFile.url);
        const arrayBuffer = await response.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        
        const rawData = audioBuffer.getChannelData(0);
        const samples = 200;
        const blockSize = Math.floor(rawData.length / samples);
        const filteredData = [];
        
        for (let i = 0; i < samples; i++) {
          let sum = 0;
          for (let j = 0; j < blockSize; j++) {
            sum += Math.abs(rawData[i * blockSize + j]);
          }
          filteredData.push(sum / blockSize);
        }
        
        const gradient = ctx.createLinearGradient(0, 0, 0, h);
        gradient.addColorStop(0, 'rgba(139, 92, 246, 0.8)');
        gradient.addColorStop(0.5, 'rgba(139, 92, 246, 0.4)');
        gradient.addColorStop(1, 'rgba(139, 92, 246, 0.1)');
        ctx.fillStyle = gradient;
        
        const barWidth = width / samples;
        const max = Math.max(...filteredData);
        
        for (let i = 0; i < samples; i++) {
          const x = i * barWidth;
          const amplitude = filteredData[i] / max;
          const barHeight = amplitude * h * 0.9;
          const y = (h - barHeight) / 2;
          ctx.fillRect(x, y, barWidth * 0.8, barHeight);
        }
      } catch (err) {
        console.error('Failed to draw waveform:', err);
      }
    };

    drawWaveform();

    return () => {
      audioContextRef.current?.close();
    };
  }, [audioFile]);

  return (
    <Box style={{ background: 'linear-gradient(180deg, rgba(13, 17, 23, 0.8) 0%, rgba(33, 38, 45, 0.6) 100%)', backdropFilter: 'blur(20px)', border: '1px solid rgba(139, 92, 246, 0.2)', borderRadius: '20px', padding: '32px', boxShadow: '0 8px 32px rgba(0, 0, 0, 0.6)' }}>
      <Stack gap="md">
        <Group justify="space-between" align="center">
          <Group gap="sm">
            <IconWaveSquare size={24} color="#8B5CF6" />
            <Text size="lg" fw={600} c="white">Audio Waveform</Text>
          </Group>
          {audioFile && (
            <Group gap="xs">
              <Badge variant="light" color="violet" size="sm">{audioFile.duration.toFixed(1)}s</Badge>
              {selectedStyle && <Badge variant="filled" color="violet" size="sm">{selectedStyle}</Badge>}
            </Group>
          )}
        </Group>

        {audioFile ? (
          <Box style={{ height: 180, background: 'linear-gradient(180deg, rgba(139, 92, 246, 0.05) 0%, rgba(139, 92, 246, 0.02) 100%)', borderRadius: '16px', padding: '24px', border: '1px solid rgba(139, 92, 246, 0.2)' }}>
            <canvas ref={canvasRef} width={1200} height={180} style={{ width: '100%', height: '100%', display: 'block' }} />
          </Box>
        ) : (
          <Box style={{ height: 180, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(139, 92, 246, 0.03)', border: '2px dashed rgba(139, 92, 246, 0.3)', borderRadius: '16px' }}>
            <Stack align="center" gap="xs">
              <IconMusic size={48} color="rgba(139, 92, 246, 0.4)" />
              <Text size="sm" c="dimmed">Upload audio to visualize waveform</Text>
            </Stack>
          </Box>
        )}
      </Stack>
    </Box>
  );
}
