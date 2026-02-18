import { useEffect, useState } from 'react';
import { Stack, Select, Slider, Switch, Text, Box, Group } from '@mantine/core';
import { IconPalette } from '@tabler/icons-react';
import { useStore } from '../store';
import { api, Style } from '../services/api';

export function StylePanel() {
  const [styles, setStyles] = useState<Style[]>([]);
  const {
    selectedStyle,
    setSelectedStyle,
    intensity,
    setIntensity,
    pitchShift,
    setPitchShift,
    tempoScale,
    setTempoScale,
    vibrato,
    setVibrato,
    swing,
    setSwing,
    autoMode,
    setAutoMode,
  } = useStore();

  useEffect(() => {
    api.getStyles().then(setStyles);
  }, []);

  return (
    <Stack gap="xl">
      <Box>
        <Group gap="xs" mb="md">
          <IconPalette size={20} color="#8B5CF6" />
          <Text size="sm" fw={600} tt="uppercase" c="dimmed" style={{ letterSpacing: '0.05em' }}>
            Style & Parameters
          </Text>
        </Group>

        <Select
          label={<Text size="sm" fw={500} mb={4}>Target Style</Text>}
          placeholder="Select musical style"
          data={styles.map(s => ({ 
            value: s.id, 
            label: `${s.icon} ${s.name}` 
          }))}
          value={selectedStyle}
          onChange={setSelectedStyle}
          size="md"
          color="violet"
        />
      </Box>

      <Box>
        <Text size="sm" fw={500} mb="md">Intensity</Text>
        <Slider
          value={intensity}
          onChange={setIntensity}
          min={0}
          max={1}
          step={0.01}
          marks={[{ value: 0, label: '0%' }, { value: 0.5, label: '50%' }, { value: 1, label: '100%' }]}
          color="violet"
          size="md"
        />
      </Box>

      <Box>
        <Text size="sm" fw={500} mb="md">Pitch Shift</Text>
        <Slider
          value={pitchShift}
          onChange={setPitchShift}
          min={-12}
          max={12}
          step={1}
          marks={[{ value: -12, label: '-12' }, { value: 0, label: '0' }, { value: 12, label: '+12' }]}
          color="violet"
          size="md"
        />
      </Box>

      <Box>
        <Text size="sm" fw={500} mb="md">Tempo</Text>
        <Slider
          value={tempoScale}
          onChange={setTempoScale}
          min={0.5}
          max={2.0}
          step={0.1}
          marks={[{ value: 0.5, label: '0.5x' }, { value: 1.0, label: '1x' }, { value: 2.0, label: '2x' }]}
          color="violet"
          size="md"
        />
      </Box>

      <Stack gap="sm">
        <Switch
          label="Vibrato"
          checked={vibrato}
          onChange={(e) => setVibrato(e.currentTarget.checked)}
          color="violet"
          size="md"
        />
        <Switch
          label="Swing"
          checked={swing}
          onChange={(e) => setSwing(e.currentTarget.checked)}
          color="violet"
          size="md"
        />
        <Switch
          label="Auto Mode"
          checked={autoMode}
          onChange={(e) => setAutoMode(e.currentTarget.checked)}
          color="violet"
          size="md"
        />
      </Stack>
    </Stack>
  );
}
