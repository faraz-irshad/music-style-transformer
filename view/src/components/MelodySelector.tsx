import { Select, Text, Box, Group } from '@mantine/core';
import { IconMusic } from '@tabler/icons-react';
import { useStore } from '../store';

export function MelodySelector() {
  const { index, selectedMelodyId, setSelectedMelodyId } = useStore();

  if (!index || index._empty || index.melodies.length === 0) return null;

  const data = index.melodies.map((m) => ({
    value: m.melody_id,
    label: m.melody_id.replace(/_/g, ' '),
  }));

  return (
    <Box>
      <Group gap="xs" mb="xs">
        <IconMusic size={16} color="#8B5CF6" />
        <Text size="sm" fw={600} c="dimmed" tt="uppercase" style={{ letterSpacing: '0.05em' }}>
          Select Melody
        </Text>
      </Group>
      <Select
        data={data}
        value={selectedMelodyId}
        onChange={setSelectedMelodyId}
        placeholder="Choose a test melody…"
        size="sm"
        styles={{ input: { background: 'rgba(33,38,45,0.8)' } }}
      />
    </Box>
  );
}
