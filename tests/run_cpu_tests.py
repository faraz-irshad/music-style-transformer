import sys
sys.path.insert(0, '.')
from layer_a.generate import (
    _key_from_melody, generate_accompaniment, TWINKLE_MELODY, Note
)
from data.skyline import skyline, Note as SNote

assert _key_from_melody([60,62,64,65,67,69,71,72]) == 0
assert _key_from_melody([67,69,71,72,74,76,78,79]) == 7
assert _key_from_melody([]) == 0
print("key detection: ok")

for style in ("baroque", "romantic"):
    acc = generate_accompaniment(TWINKLE_MELODY, style, seed=42)
    assert len(acc) > 0
    for n in acc:
        assert n.end > n.start, f"bad duration {n}"
        assert 0 <= n.velocity <= 127, f"bad vel {n}"
print("accompaniment generation: ok")

b = generate_accompaniment(TWINKLE_MELODY, "baroque", seed=42)
r = generate_accompaniment(TWINKLE_MELODY, "romantic", seed=42)
assert sorted(n.pitch for n in b) != sorted(n.pitch for n in r)
print("styles differ: ok")

b2 = generate_accompaniment(TWINKLE_MELODY, "baroque", seed=42)
assert [(n.start, n.pitch) for n in b] == [(n.start, n.pitch) for n in b2]
print("determinism: ok")

mean_b = sum(n.end - n.start for n in b) / len(b)
mean_r = sum(n.end - n.start for n in r) / len(r)
assert mean_b < mean_r, f"baroque {mean_b:.3f} not < romantic {mean_r:.3f}"
print(f"staccato vs legato: baroque={mean_b:.3f}s romantic={mean_r:.3f}s ok")

r2 = generate_accompaniment(TWINKLE_MELODY, "romantic", seed=99)
assert [n.start for n in r] != [n.start for n in r2]
print("different seeds differ: ok")

melody_max = max(p for _, _, p in TWINKLE_MELODY)
above = sum(1 for n in b if n.pitch > melody_max)
assert above / len(b) < 0.1, f"{above}/{len(b)} notes above melody"
print("accompaniment below melody: ok")

sorted_r = sorted(r, key=lambda n: n.start)
half = len(sorted_r) // 2
first_vel = sum(n.velocity for n in sorted_r[:half]) / half
second_vel = sum(n.velocity for n in sorted_r[half:]) / (len(sorted_r) - half)
assert second_vel > first_vel, f"rising dynamics failed: {first_vel:.1f} {second_vel:.1f}"
print(f"rising dynamics: {first_vel:.1f} -> {second_vel:.1f} ok")

assert skyline([]) == []
res = skyline([SNote(0.0, 1.0, 60), SNote(0.0, 1.0, 67)])
assert len(res) == 1 and res[0].pitch == 67
res2 = skyline([SNote(0.0, 0.5, 60), SNote(0.5, 1.0, 64), SNote(1.0, 1.5, 67)])
assert [n.pitch for n in res2] == [60, 64, 67]
mel_notes = [SNote(s, e, p) for s, e, p in TWINKLE_MELODY]
sky = skyline(mel_notes)
assert [n.pitch for n in sky] == [p for _, _, p in TWINKLE_MELODY]
print("skyline: ok")

print("\nALL TESTS PASSED")
