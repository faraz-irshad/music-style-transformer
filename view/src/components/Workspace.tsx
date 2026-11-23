import { useEffect } from 'react';
import { Box, Stack, Text, Loader, Center } from '@mantine/core';
import { Header } from './Header';
import { MelodySelector } from './MelodySelector';
import { ComparisonGrid } from './ComparisonGrid';
import { ABToggle } from './ABToggle';
import { MethodPanel } from './MethodPanel';
import { EmptyState } from './EmptyState';
import { useStore } from '../store';
import { loadDemoIndex } from '../services/api';

export function Workspace() {
  const {
    index, loading, error,
    selectedMelodyId, setSelectedMelodyId,
    setIndex, setLoading, setError,
  } = useStore();

  useEffect(() => {
    loadDemoIndex()
      .then((data) => {
        setIndex(data);
        if (data.melodies.length > 0) {
          setSelectedMelodyId(data.melodies[0].melody_id);
        }
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  const selectedMelody = index?.melodies.find((m) => m.melody_id === selectedMelodyId);
  const isEmpty = !index || index._empty || index.melodies.length === 0;

  return (
    <Box style={{
      minHeight: '100vh',
      background: 'linear-gradient(180deg, #0D1117 0%, #161B22 60%, #21262D 100%)',
      display: 'flex',
      flexDirection: 'column',
    }}>
      <Header />

      <Box style={{ flex: 1, padding: '32px 24px', maxWidth: 1200, margin: '0 auto', width: '100%' }}>
        {loading && (
          <Center mt="xl">
            <Loader color="violet" />
          </Center>
        )}

        {error && (
          <Text c="red" mt="md">Error loading demo index: {error}</Text>
        )}

        {!loading && !error && (
          <Stack gap="xl">
            {/* Melody selector */}
            {!isEmpty && <MelodySelector />}

            {/* Empty state */}
            {isEmpty && <EmptyState />}

            {/* Comparison grid */}
            {!isEmpty && selectedMelody && index && (
              <Box>
                <Text size="sm" fw={600} c="dimmed" tt="uppercase"
                  style={{ letterSpacing: '0.05em' }} mb="sm">
                  Comparison — {selectedMelody.melody_id.replace(/_/g, ' ')}
                </Text>
                <ComparisonGrid
                  outputs={selectedMelody.outputs}
                  layers={index.layers}
                />
              </Box>
            )}

            {/* A/B toggle */}
            {!isEmpty && selectedMelody && (
              <ABToggle outputs={selectedMelody.outputs} />
            )}

            {/* Method and honesty panel */}
            <MethodPanel summary={index?.summary ?? {}} />
          </Stack>
        )}
      </Box>
    </Box>
  );
}
