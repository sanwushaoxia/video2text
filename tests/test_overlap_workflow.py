import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

import numpy as np
import soundfile as sf

from cli.correct_overlap import apply_corrections
from video2text.diarization import DiarizationResult, DiarizationTurn, fuse_diarization, detect_overlap_regions
from video2text.overlap_tracking import Window, track_windows
from video2text.separate import _write_residual_background


class OverlapWorkflowTest(unittest.TestCase):
    def test_audio_continuity_resolves_a_locally_ambiguous_window(self):
        sr = 1000
        t = np.arange(200) / sr
        a = np.sin(2 * np.pi * 25 * t)
        b = np.cos(2 * np.pi * 10 * t)
        w1 = Window(20, 100, 0, (a[:160], b[:160]), {})
        w2 = Window(100, 160, 60, (b[60:], a[60:]), {})
        embedder = Mock()
        embedder.embed_waveform.side_effect = [
            np.array([1., 0.]), np.array([0., 1.]),
            np.array([.7, .7]), np.array([.7, .7]),
        ]
        track_windows(
            [w1, w2], sr, ["a", "b"], {"a": np.array([1., 0.]), "b": np.array([0., 1.])},
            {}, embedder, .35,
        )
        self.assertEqual(w1.report["source_speakers"], ["a", "b"])
        self.assertEqual(w2.report["source_speakers"], ["b", "a"])

    def test_joint_aligns_speaker_ids_and_sends_disagreement_to_review(self):
        def result(turns):
            turns = [DiarizationTurn(*t) for t in turns]
            return DiarizationResult(turns, detect_overlap_regions(turns), "test")

        mix = result([(0., 3., "a"), (2., 5., "b")])
        vocals = result([(0., 3.2, "y"), (2.4, 5., "x")])
        joint = fuse_diarization(mix, vocals)
        self.assertEqual(joint.turns, mix.turns)
        self.assertEqual([(r.start, r.end, r.speaker_ids) for r in joint.overlap_regions],
                         [(2.4, 3., ["a", "b"])])
        self.assertEqual([(r.start, r.end, r.reason) for r in joint.overlap_review_regions],
                         [(2., 2.4, "joint_disagreement"), (3., 3.2, "joint_disagreement")])

    def test_manual_correction_only_writes_the_selected_sample_range(self):
        with TemporaryDirectory() as temp:
            src = Path(temp)
            sr = 44100
            originals = {name: np.full(220, value, dtype=np.float32)
                         for name, value in (("a", .1), ("b", .2), ("review", .4))}
            for name, signal in originals.items():
                sf.write(src / f"{name}.wav", signal, sr, subtype="FLOAT")
            source = np.column_stack((np.full(40, .3), np.full(40, .7))).astype(np.float32)
            preview = src / "preview.wav"
            sf.write(preview, source, sr, subtype="FLOAT")
            report = src / "report.json"
            report.write_text(json.dumps({
                "speaker_tracks": {s: str(src / f"{s}.wav") for s in ("a", "b")},
                "dialogue_review": str(src / "review.wav"),
                "overlap_bss": {"sample_rate": sr, "regions": [{
                    "id": "0:0", "status": "review", "start_sample": 80, "end_sample": 120,
                    "preview": str(preview), "backend": "sepformer",
                }]},
            }))
            with self.assertRaises(ValueError):
                apply_corrections(report, {"0:0": ["a", "a"]}, src / "invalid")
            self.assertFalse((src / "invalid").exists())
            updated = apply_corrections(report, {"0:0": ["b", "a"]}, src / "edited")
            metadata = json.loads(updated.read_text())
            self.assertEqual(metadata["overlap_bss"]["regions"][0]["assignment"], "manual")
            for name, expected in (("a", source[:, 1]), ("b", source[:, 0]), ("review", 0.)):
                result, rate = sf.read(updated.parent / f"{name}.wav", dtype="float32")
                self.assertEqual(rate, sr)
                np.testing.assert_array_equal(result[80:120], expected)
                np.testing.assert_array_equal(result[:80], originals[name][:80])
                np.testing.assert_array_equal(result[120:], originals[name][120:])
                np.testing.assert_array_equal(sf.read(src / f"{name}.wav", dtype="float32")[0],
                                              originals[name])

    def test_residual_background_preserves_timeline_and_stereo_effects(self):
        with TemporaryDirectory() as temp:
            path = Path(temp)
            sr = 44100
            background = np.zeros((sr, 2), dtype=np.float32)
            background[250, 0] = .25
            background[650, 1] = -.5
            dialogue = np.full((sr, 2), .125, dtype=np.float32)
            sf.write(path / "mix.wav", background + dialogue, sr, subtype="FLOAT")
            sf.write(path / "vocals.wav", dialogue, sr, subtype="FLOAT")
            _write_residual_background(path / "mix.wav", path / "vocals.wav", path / "out.wav")
            actual, rate = sf.read(path / "out.wav", dtype="float32")
            self.assertEqual(rate, sr)
            np.testing.assert_array_equal(actual, background)


if __name__ == "__main__":
    unittest.main()
