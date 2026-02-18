import { Stack, Button, Text, Group, FileButton, Box } from '@mantine/core';
import { IconUpload, IconMicrophone, IconFileMusic } from '@tabler/icons-react';
import { useStore } from '../store';
import { notifications } from '@mantine/notifications';

export function InputPanel() {
  const { audioFile, setAudioFile } = useStore();

  const handleUpload = async (file: File | null) => {
    if (!file) return;
    const validTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/midi'];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(wav|mp3|midi?)$/i)) {
      notifications.show({ title: 'Invalid file', message: 'Please upload WAV, MP3, or MIDI', color: 'red' });
      return;
    }
    const url = URL.createObjectURL(file);
    const audio = new Audio(url);
    audio.addEventListener('loadedmetadata', () => {
      setAudioFile({ file, url, duration: audio.duration });
      notifications.show({ title: 'Success', message: `${file.name} loaded`, color: 'green' });
    });
    audio.addEventListener('error', () => {
      notifications.show({ title: 'Error', message: 'Failed to load audio', color: 'red' });
    });
  };

  return (
    <Stack gap="lg">
      <Group gap="xs">
        <IconFileMusic size={20} color="#8B5CF6" />
        <Text size="sm" fw={600} tt="uppercase" c="dimmed" style={{ letterSpacing: '0.05em' }}>
          Input Audio
        </Text>
      </Group>
      
      <FileButton onChange={handleUpload} accept="audio/wav,audio/mp3,audio/mpeg,audio/midi">
        {(props) => (
          <Button
            {...props}
            leftSection={<IconUpload size={18} />}
            variant="light"
            size="lg"
            fullWidth
            color="violet"
          >
            Upload Audio File
          </Button>
        )}
      </FileButton>

      <Button
        leftSection={<IconMicrophone size={18} />}
        variant="outline"
        size="lg"
        fullWidth
        color="violet"
      >
        Record Audio
      </Button>

      {audioFile && (
        <Box
          p="md"
          style={{
            background: 'rgba(139, 92, 246, 0.08)',
            border: '1px solid rgba(139, 92, 246, 0.2)',
            borderRadius: '12px',
          }}
        >
          <Stack gap="xs">
            <Text size="xs" fw={500} c="violet">
              Loaded File
            </Text>
            <Text size="sm" truncate c="white">
              {audioFile.file.name}
            </Text>
            <Text size="xs" c="dimmed">
              {audioFile.duration.toFixed(2)}s • {(audioFile.file.size / 1024 / 1024).toFixed(2)} MB
            </Text>
          </Stack>
        </Box>
      )}
    </Stack>
  );
}
