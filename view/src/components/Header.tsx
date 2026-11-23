import { Group, Box, Text, Anchor } from '@mantine/core';
import { IconBrandGithub } from '@tabler/icons-react';

export function Header() {
  return (
    <Box
      style={{
        background: 'rgba(13, 17, 23, 0.97)',
        backdropFilter: 'blur(12px)',
        borderBottom: '1px solid rgba(139, 92, 246, 0.2)',
        position: 'sticky',
        top: 0,
        zIndex: 100,
      }}
    >
      <Group justify="space-between" p="md" px="xl">
        <Box>
          <Text size="lg" fw={700} style={{
            background: 'linear-gradient(135deg, #8B5CF6 0%, #6366F1 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}>
            Style-Conditioned Piano Accompaniment Generation
          </Text>
          <Text size="xs" c="dimmed">
            BSc Thesis · ELTE Budapest · Fixed melody in, Baroque and Romantic accompaniment out.{' '}
            <Text span c="orange.4" size="xs">Not full style transfer.</Text>
          </Text>
        </Box>
        <Anchor
          href="https://github.com/faraz-irshad/music-style-transformer"
          target="_blank"
          c="dimmed"
        >
          <IconBrandGithub size={20} />
        </Anchor>
      </Group>
    </Box>
  );
}
