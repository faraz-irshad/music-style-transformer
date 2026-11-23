import { Box, Text, Badge, Stack, Group, SimpleGrid } from '@mantine/core';
import { AudioPlayer } from './AudioPlayer';
import type { Output, Metrics } from '../store';

const STYLE_COLORS: Record<string, string> = {
  baroque: 'indigo',
  romantic: 'pink',
  none: 'gray',
};

const LAYER_BADGE_COLORS: Record<string, string> = {
  base_amt: 'gray',
  layer_a_scripted: 'orange',
  layer_c_lora: 'violet',
};

function MetricsDisplay({ m }: { m: Metrics }) {
  const items: [string, string][] = [];
  if (m.clf_prob_baroque != null)
    items.push(['P(baroque)', m.clf_prob_baroque.toFixed(2)]);
  if (m.clf_prob_romantic != null)
    items.push(['P(romantic)', m.clf_prob_romantic.toFixed(2)]);
  if (m.skyline_dom != null)
    items.push(['skyline dom', m.skyline_dom.toFixed(2)]);
  if (m.clash_rate != null)
    items.push(['clash rate', m.clash_rate.toFixed(2)]);
  if (!items.length) return null;
  return (
    <Box mt={6}>
      {items.map(([k, v]) => (
        <Group key={k} justify="space-between" gap={4}>
          <Text size="xs" c="dimmed">{k}</Text>
          <Text size="xs" c="white" fw={500}>{v}</Text>
        </Group>
      ))}
    </Box>
  );
}

function OutputCell({ output }: { output: Output | undefined; style: string }) {
  if (!output) {
    return (
      <Box p="sm" style={{ background: 'rgba(255,255,255,0.03)', borderRadius: 8, minHeight: 80 }}>
        <Text size="xs" c="dimmed">—</Text>
      </Box>
    );
  }
  return (
    <Box p="sm" style={{
      background: 'rgba(33,38,45,0.7)',
      border: '1px solid rgba(139,92,246,0.15)',
      borderRadius: 8,
    }}>
      <Group gap={6} mb={6} wrap="wrap">
        <Badge size="xs" color={LAYER_BADGE_COLORS[output.layer] ?? 'gray'} variant="light">
          {output.layer_label}
        </Badge>
        <Badge size="xs" color={STYLE_COLORS[output.style] ?? 'gray'} variant="dot">
          {output.style}
        </Badge>
      </Group>
      {output.audio_url ? (
        <AudioPlayer url={output.audio_url} />
      ) : (
        <Text size="xs" c="dimmed" fs="italic">No audio — run the pipeline</Text>
      )}
      <MetricsDisplay m={output.metrics} />
    </Box>
  );
}

interface Props {
  outputs: Output[];
  layers: { id: string; label: string }[];
}

export function ComparisonGrid({ outputs, layers }: Props) {
  const styles = ['baroque', 'romantic'];

  return (
    <Box>
      {/* Column headers */}
      <SimpleGrid cols={3} mb="xs">
        <Box />
        {styles.map((s) => (
          <Text key={s} size="sm" fw={600} c={s === 'baroque' ? 'indigo.3' : 'pink.3'}
            tt="capitalize" ta="center">
            {s}
          </Text>
        ))}
      </SimpleGrid>

      {/* Rows */}
      <Stack gap="xs">
        {layers.map((layer) => {
          const baroque = outputs.find(
            (o) => o.layer === layer.id && o.style === 'baroque'
          );
          const romantic = outputs.find(
            (o) => o.layer === layer.id && o.style === 'romantic'
          );
          // base_amt has style=none — show in both columns
          const baseNone = outputs.find(
            (o) => o.layer === layer.id && o.style === 'none'
          );

          return (
            <SimpleGrid key={layer.id} cols={3} style={{ alignItems: 'start' }}>
              <Box pt="xs">
                <Text size="xs" fw={600} c="dimmed">{layer.label}</Text>
              </Box>
              <OutputCell
                output={baroque ?? (layer.id === 'base_amt' ? baseNone : undefined)}
                style="baroque"
              />
              <OutputCell
                output={romantic ?? (layer.id === 'base_amt' ? baseNone : undefined)}
                style="romantic"
              />
            </SimpleGrid>
          );
        })}
      </Stack>
    </Box>
  );
}
