import { Stack, Button, Text, Progress, Box, Group, List } from '@mantine/core';
import { IconPlayerPlay, IconDownload, IconSparkles } from '@tabler/icons-react';
import { useStore } from '../store';
import { api } from '../services/api';
import { AudioPlayer } from './AudioPlayer';
import { notifications } from '@mantine/notifications';

export function OutputPanel() {
  const {
    audioFile,
    selectedStyle,
    intensity,
    pitchShift,
    tempoScale,
    vibrato,
    swing,
    autoMode,
    isProcessing,
    progress,
    previewUrl,
    exportHistory,
    setIsProcessing,
    setProgress,
    setPreviewUrl,
    addExport,
  } = useStore();

  const canTransform = audioFile && selectedStyle && !isProcessing;

  const generatePreview = async () => {
    if (!canTransform) return;
    setIsProcessing(true);
    setProgress(0);
    try {
      const url = await api.transformPreview({ audioFile: audioFile.file, styleId: selectedStyle, intensity, pitchShift, tempoScale, vibrato, swing, autoMode }, setProgress);
      setPreviewUrl(url);
      notifications.show({ title: 'Preview Ready', message: '5-second preview generated', color: 'green' });
    } catch {
      notifications.show({ title: 'Failed', message: 'Could not generate preview', color: 'red' });
    } finally {
      setIsProcessing(false);
      setProgress(0);
    }
  };

  const exportFull = async () => {
    if (!canTransform) return;
    setIsProcessing(true);
    setProgress(0);
    try {
      const url = await api.transformFull({ audioFile: audioFile.file, styleId: selectedStyle, intensity, pitchShift, tempoScale, vibrato, swing, autoMode }, setProgress);
      const filename = `transformed_${selectedStyle}_${Date.now()}.wav`;
      addExport({ id: Date.now().toString(), timestamp: new Date(), styleName: selectedStyle, filename, url });
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.click();
      notifications.show({ title: 'Export Complete', message: filename, color: 'green' });
    } catch {
      notifications.show({ title: 'Failed', message: 'Could not export audio', color: 'red' });
    } finally {
      setIsProcessing(false);
      setProgress(0);
    }
  };

  return (
    <Stack gap="xl">
      <Box>
        <Group gap="xs" mb="md">
          <IconSparkles size={20} color="#8B5CF6" />
          <Text size="sm" fw={600} tt="uppercase" c="dimmed" style={{ letterSpacing: '0.05em' }}>
            Transform & Export
          </Text>
        </Group>

        <Button
          leftSection={<IconPlayerPlay size={20} />}
          onClick={generatePreview}
          disabled={!canTransform}
          loading={isProcessing}
          size="xl"
          fullWidth
          variant="gradient"
          gradient={{ from: 'violet', to: 'indigo', deg: 135 }}
          style={{
            height: '64px',
            fontSize: '16px',
            fontWeight: 600,
            boxShadow: canTransform ? '0 8px 24px rgba(139, 92, 246, 0.4)' : 'none',
          }}
        >
          Generate Preview (5s)
        </Button>
      </Box>

      {isProcessing && (
        <Box>
          <Progress value={progress} animated color="violet" size="lg" radius="md" style={{ boxShadow: '0 0 20px rgba(139, 92, 246, 0.3)' }} />
          <Text size="xs" c="dimmed" ta="center" mt="xs">Processing... {progress}%</Text>
        </Box>
      )}

      {previewUrl && (
        <Box p="lg" style={{ background: 'rgba(139, 92, 246, 0.08)', border: '1px solid rgba(139, 92, 246, 0.2)', borderRadius: '12px' }}>
          <Stack gap="md">
            <Text size="sm" fw={500} c="violet">Preview Player</Text>
            <AudioPlayer url={previewUrl} />
          </Stack>
        </Box>
      )}

      <Button
        leftSection={<IconDownload size={18} />}
        onClick={exportFull}
        disabled={!canTransform}
        loading={isProcessing}
        variant="light"
        size="lg"
        fullWidth
        color="violet"
      >
        Export Full Transformation
      </Button>

      {exportHistory.length > 0 && (
        <Box p="md" style={{ background: 'rgba(139, 92, 246, 0.05)', border: '1px solid rgba(139, 92, 246, 0.15)', borderRadius: '12px' }}>
          <Stack gap="sm">
            <Text size="xs" fw={600} tt="uppercase" c="dimmed" style={{ letterSpacing: '0.05em' }}>Export History</Text>
            <List size="xs" spacing={8}>
              {exportHistory.map((item) => (
                <List.Item key={item.id}>
                  <Text size="xs" truncate c="white">{item.filename}</Text>
                  <Text size="xs" c="dimmed">{item.styleName} • {item.timestamp.toLocaleTimeString()}</Text>
                </List.Item>
              ))}
            </List>
          </Stack>
        </Box>
      )}
    </Stack>
  );
}
