import { Group, ActionIcon, Text, Box } from '@mantine/core';
import { IconRotateClockwise, IconHelp, IconBrandGithub } from '@tabler/icons-react';
import { useStore } from '../store';
import { notifications } from '@mantine/notifications';

export function Header() {
  const { reset } = useStore();

  return (
    <Box
      style={{
        background: 'rgba(13, 17, 23, 0.95)',
        backdropFilter: 'blur(12px)',
        borderBottom: '1px solid rgba(139, 92, 246, 0.2)',
        position: 'sticky',
        top: 0,
        zIndex: 100,
      }}
    >
      <Group justify="space-between" p="lg" px="xl">
        <Group gap="sm">
          <Box
            style={{
              width: 40,
              height: 40,
              borderRadius: 10,
              background: 'linear-gradient(135deg, #8B5CF6 0%, #6366F1 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 20,
            }}
          >
            🎵
          </Box>
          <Box>
            <Text size="xl" fw={700} style={{ 
              background: 'linear-gradient(135deg, #8B5CF6 0%, #6366F1 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              letterSpacing: '-0.02em',
            }}>
              Music Style Transformer
            </Text>
            <Text size="xs" c="dimmed">AI-Powered Audio Research Tool</Text>
          </Box>
        </Group>
        
        <Group gap="xs">
          <ActionIcon variant="subtle" size="lg" onClick={() => reset()} style={{ transition: 'all 0.2s' }}>
            <IconRotateClockwise size={18} />
          </ActionIcon>
          <ActionIcon variant="subtle" size="lg" onClick={() => notifications.show({ title: 'Help', message: 'Upload audio → Select style → Adjust parameters → Generate preview → Export' })} style={{ transition: 'all 0.2s' }}>
            <IconHelp size={18} />
          </ActionIcon>
          <ActionIcon variant="subtle" size="lg" component="a" href="https://github.com/faraz-irshad/music-style-transformer" target="_blank" style={{ transition: 'all 0.2s' }}>
            <IconBrandGithub size={18} />
          </ActionIcon>
        </Group>
      </Group>
    </Box>
  );
}
