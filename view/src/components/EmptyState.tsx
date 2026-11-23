import { Box, Stack, Text } from '@mantine/core';
import { IconMusicOff } from '@tabler/icons-react';

export function EmptyState() {
  return (
    <Box style={{
      minHeight: 300,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      border: '2px dashed rgba(139,92,246,0.3)',
      borderRadius: 16,
    }}>
      <Stack align="center" gap="sm">
        <IconMusicOff size={48} color="rgba(139,92,246,0.4)" />
        <Text size="md" c="dimmed" fw={500}>No generated results yet</Text>
        <Text size="xs" c="dimmed" ta="center" maw={360}>
          Run the offline pipeline to generate results:
        </Text>
        <Box p="sm" style={{
          background: 'rgba(0,0,0,0.4)',
          borderRadius: 8,
          fontFamily: 'monospace',
        }}>
          <Text size="xs" c="violet">python generate.py --config config.yaml</Text>
          <Text size="xs" c="violet">python evaluate.py --config config.yaml</Text>
          <Text size="xs" c="violet">python scripts/build_demo_index.py --config config.yaml</Text>
        </Box>
        <Text size="xs" c="dimmed" fs="italic">
          Generation runs on a Colab GPU — see PROGRESS.md for instructions.
        </Text>
      </Stack>
    </Box>
  );
}
