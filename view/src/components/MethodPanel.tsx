import { Box, Text, Stack, Group, Badge } from '@mantine/core';

interface Props {
  summary: Record<string, unknown>;
}

function fmt(v: unknown): string {
  if (v == null) return '—';
  if (typeof v === 'number') return v.toFixed(3);
  if (typeof v === 'object') return JSON.stringify(v, null, 2);
  return String(v);
}

export function MethodPanel({ summary }: Props) {
  const hasResults = summary && Object.keys(summary).length > 0;

  return (
    <Box p="lg" style={{
      background: 'rgba(33,38,45,0.6)',
      border: '1px solid rgba(139,92,246,0.15)',
      borderRadius: 12,
    }}>
      <Text size="sm" fw={700} c="violet" mb="md" tt="uppercase"
        style={{ letterSpacing: '0.05em' }}>
        Method &amp; Honesty
      </Text>

      <Stack gap="xs">
        <Text size="xs" c="dimmed">
          A fixed melody goes in. A new piano accompaniment is generated around it in
          Baroque or Romantic style. The melody is never changed.
          This is <Text span c="orange.4" size="xs" fw={600}>not full style transfer</Text> —
          no existing arrangement is restyled.
        </Text>

        <Group gap="xs" mt="xs">
          <Badge size="xs" color="gray" variant="outline">base_amt</Badge>
          <Text size="xs" c="dimmed">Pretrained AMT (Lakh MIDI), melody as controls only. Neutral baseline.</Text>
        </Group>
        <Group gap="xs">
          <Badge size="xs" color="orange" variant="outline">layer_a_scripted</Badge>
          <Text size="xs" c="dimmed">Rule-based. NOT a learned model. Guaranteed contrast.</Text>
        </Group>
        <Group gap="xs">
          <Badge size="xs" color="violet" variant="outline">layer_c_lora</Badge>
          <Text size="xs" c="dimmed">LoRA adapter (r=8) fine-tuned on MAESTRO Baroque/Romantic.</Text>
        </Group>

        {hasResults && (
          <Box mt="sm" p="sm" style={{
            background: 'rgba(0,0,0,0.3)',
            borderRadius: 8,
            fontFamily: 'monospace',
          }}>
            <Text size="xs" c="violet" mb={4}>Evaluation results (from evaluation_summary.json):</Text>
            {Object.entries(summary).map(([k, v]) => (
              <Text key={k} size="xs" c="dimmed">
                <Text span c="white">{k}</Text>: {fmt(v)}
              </Text>
            ))}
          </Box>
        )}

        {!hasResults && (
          <Text size="xs" c="dimmed" fs="italic" mt="xs">
            No evaluation results yet — run the pipeline first.
          </Text>
        )}
      </Stack>
    </Box>
  );
}
