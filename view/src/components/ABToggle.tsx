import { useRef, useState } from 'react';
import { Box, Group, Button, Text, SegmentedControl } from '@mantine/core';
import { IconPlayerPlay, IconPlayerStop } from '@tabler/icons-react';
import type { Output } from '../store';

interface Props {
  outputs: Output[];
}

export function ABToggle({ outputs }: Props) {
  const [style, setStyle] = useState<'baroque' | 'romantic'>('baroque');
  const [playing, setPlaying] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const baroque = outputs.find((o) => o.style === 'baroque' && o.layer === 'layer_a_scripted')
    ?? outputs.find((o) => o.style === 'baroque');
  const romantic = outputs.find((o) => o.style === 'romantic' && o.layer === 'layer_a_scripted')
    ?? outputs.find((o) => o.style === 'romantic');

  const current = style === 'baroque' ? baroque : romantic;

  const toggle = () => {
    if (!current?.audio_url) return;
    if (playing) {
      audioRef.current?.pause();
      setPlaying(false);
    } else {
      const audio = new Audio(current.audio_url);
      audioRef.current = audio;
      audio.play();
      audio.onended = () => setPlaying(false);
      setPlaying(true);
    }
  };

  const switchStyle = (s: string) => {
    audioRef.current?.pause();
    setPlaying(false);
    setStyle(s as 'baroque' | 'romantic');
  };

  if (!baroque?.audio_url && !romantic?.audio_url) return null;

  return (
    <Box p="md" style={{
      background: 'rgba(33,38,45,0.7)',
      border: '1px solid rgba(139,92,246,0.2)',
      borderRadius: 12,
    }}>
      <Text size="sm" fw={600} c="dimmed" mb="sm" tt="uppercase"
        style={{ letterSpacing: '0.05em' }}>
        A/B Toggle — same melody, different style
      </Text>
      <Group gap="md" align="center">
        <SegmentedControl
          value={style}
          onChange={switchStyle}
          data={[
            { label: 'Baroque', value: 'baroque' },
            { label: 'Romantic', value: 'romantic' },
          ]}
          color="violet"
          size="sm"
        />
        <Button
          leftSection={playing ? <IconPlayerStop size={14} /> : <IconPlayerPlay size={14} />}
          onClick={toggle}
          disabled={!current?.audio_url}
          variant={playing ? 'filled' : 'light'}
          color="violet"
          size="sm"
        >
          {playing ? 'Stop' : 'Play'}
        </Button>
        <Text size="xs" c="dimmed">{current?.layer_label ?? '—'}</Text>
      </Group>
    </Box>
  );
}
