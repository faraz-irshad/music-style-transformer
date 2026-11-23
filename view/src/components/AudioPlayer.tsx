import { useRef, useState, useEffect } from 'react';
import { Group, ActionIcon, Slider, Box, Text } from '@mantine/core';
import { IconPlayerPlay, IconPlayerPause } from '@tabler/icons-react';

interface AudioPlayerProps {
  url: string;
  syncRef?: React.MutableRefObject<HTMLAudioElement | null>;
}

export function AudioPlayer({ url, syncRef }: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    if (syncRef) syncRef.current = audioRef.current;
  }, [syncRef]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const onTime = () => setCurrentTime(audio.currentTime);
    const onMeta = () => setDuration(audio.duration);
    const onEnd = () => setIsPlaying(false);
    audio.addEventListener('timeupdate', onTime);
    audio.addEventListener('loadedmetadata', onMeta);
    audio.addEventListener('ended', onEnd);
    return () => {
      audio.removeEventListener('timeupdate', onTime);
      audio.removeEventListener('loadedmetadata', onMeta);
      audio.removeEventListener('ended', onEnd);
    };
  }, []);

  const toggle = () => {
    const audio = audioRef.current;
    if (!audio) return;
    isPlaying ? audio.pause() : audio.play();
    setIsPlaying(!isPlaying);
  };

  const seek = (v: number) => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = v;
    setCurrentTime(v);
  };

  const fmt = (t: number) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`;

  return (
    <Box>
      <audio ref={audioRef} src={url} />
      <Group gap="sm" wrap="nowrap">
        <ActionIcon variant="filled" size="md" onClick={toggle} color="violet"
          style={{ boxShadow: isPlaying ? '0 0 12px rgba(139,92,246,0.5)' : 'none' }}>
          {isPlaying ? <IconPlayerPause size={14} /> : <IconPlayerPlay size={14} />}
        </ActionIcon>
        <Box style={{ flex: 1 }}>
          <Slider value={currentTime} max={duration || 100} onChange={seek}
            color="violet" size="xs" />
        </Box>
        <Text size="xs" c="dimmed" style={{ minWidth: 72, textAlign: 'right' }}>
          {fmt(currentTime)} / {fmt(duration)}
        </Text>
      </Group>
    </Box>
  );
}
