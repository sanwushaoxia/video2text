import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from video2text.diarization import (
    DiarizationResult,
    DiarizationTurn,
    OverlapRegion,
    _threshold_overlap_posterior,
    detect_overlap_regions,
)
from video2text.overlap_separation import (
    _align_sources,
    _match_sources,
    apply_overlap_separation,
)
from video2text.speaker_stems import build_speaker_stems, split_vocals_by_speaker


class OverlapSeparationTest(unittest.TestCase):
    def test_aligns_eight_khz_output_and_calibrates_opposite_polarities(self):
        sr, model_sr = 44100, 8000
        t = np.arange(sr) / sr
        tm = np.arange(model_sr) / model_sr
        a = (0.08 * np.sin(2 * np.pi * 170 * t)).astype(np.float32)
        b = (0.04 * np.sin(2 * np.pi * 290 * t)).astype(np.float32)
        backend = (-np.sin(2 * np.pi * 170 * tm), np.sin(2 * np.pi * 290 * tm))
        x, y = _align_sources(backend, model_sr, a + b, sr)
        self.assertEqual((len(x), len(y)), (sr, sr))
        np.testing.assert_allclose(x[100:-100], a[100:-100], atol=0.0005)
        np.testing.assert_allclose(y[100:-100], b[100:-100], atol=0.0005)
        with self.assertRaisesRegex(RuntimeError, "时长"):
            _align_sources((backend[0][:-100], backend[1]), model_sr, a + b, sr)

    def test_assignment_uses_both_speakers_when_nearest_is_the_same(self):
        profiles = {"a": np.array([1., 0.]), "b": np.array([0., 1.])}

        class Embedder:
            def embed_waveform(self, samples, sr):
                return samples

        result = _match_sources(
            [np.array([0.95, 0.31]), np.array([0.75, 0.66])],
            44100, ["a", "b"], profiles, Embedder(), 0.35,
        )
        self.assertEqual(result, ["a", "b"])
        ambiguous = [np.array([0.71, 0.71])] * 2
        self.assertIsNone(_match_sources(ambiguous, 44100, ["a", "b"], profiles, Embedder(), 0.35))

    def test_timeline_preserves_changes_in_simultaneous_speaker_count(self):
        turns = [DiarizationTurn(0, 4, "a"), DiarizationTurn(1, 4, "b"),
                 DiarizationTurn(2, 3, "c"), DiarizationTurn(2.01, 2.05, "a")]
        self.assertEqual(detect_overlap_regions(turns), [
            OverlapRegion(1, 2, ["a", "b"]),
            OverlapRegion(2, 3, ["a", "b", "c"]),
            OverlapRegion(3, 4, ["a", "b"]),
        ])

    def test_posterior_uses_contiguous_high_confidence_runs(self):
        scores = np.array([0.9, 0.2, 0.41, 0.5, 0.6, 0.1])
        regions = _threshold_overlap_posterior(scores, 0.0, 0.1, 0.4, 0.2)
        self.assertEqual(regions, [OverlapRegion(0.2, 0.5)])

    def test_mix_diarization_and_review_posterior_preserve_vocal_audio(self):
        sr = 44100
        with TemporaryDirectory() as temp:
            path = Path(temp)
            vocals = path / "vocals.wav"
            mixture = path / "mix.wav"
            signal = np.full(sr, 0.125, dtype=np.float32)
            sf.write(vocals, signal, sr, subtype="FLOAT")
            sf.write(mixture, signal * 2, sr, subtype="FLOAT")
            diar = DiarizationResult(
                [DiarizationTurn(0, 1, "a")], [], "test", ["a"],
                overlap_review_regions=[OverlapRegion(0.25, 0.5)],
            )
            with patch("video2text.speaker_stems.run_diarization", return_value=diar) as run:
                _, stems = split_vocals_by_speaker(
                    vocals, path, "sample", path / "report.json", split_mode="diarize_bss",
                    original_audio_path=mixture, diarization_source="mix", use_embedding=False,
                    overlap_review_threshold=0.4,
                )
            self.assertEqual(run.call_args.args[0], mixture)
            self.assertEqual(run.call_args.kwargs["overlap_review_threshold"], 0.4)
            speaker = sf.read(stems["a"])[0]
            review = sf.read(path / "sample_dialogue_review.wav")[0]
            np.testing.assert_array_equal(speaker + review, signal)
            np.testing.assert_array_equal(speaker[sr // 4:sr // 2], 0)
            np.testing.assert_array_equal(review[sr // 4:sr // 2], 0.125)
            report = json.loads((path / "report.json").read_text())
            self.assertEqual(report["diarization_source"], "mix")
            self.assertEqual(report["overlap_bss"]["regions"][0]["reason"], "overlap_posterior")

    def test_partial_failure_in_long_overlap_continues_at_next_window(self):
        sr = 1000
        audio = np.ones(3200, dtype=np.float32)
        assignments = [{"start": 0, "end": 3.2, "speaker_id": s} for s in ("a", "b")]
        profiles = {"a": np.array([1., 0.]), "b": np.array([0., 1.])}
        lengths = []

        def separate(samples, sr, work_dir, *, backend):
            lengths.append(len(samples))
            if len(lengths) == 2:
                raise RuntimeError("inference unavailable")
            return samples * 0.25, samples * 0.75, "sepformer"

        with TemporaryDirectory() as temp, patch(
            "video2text.overlap_separation.separate_overlap_2spk", side_effect=separate
        ), patch("video2text.overlap_separation._match_sources", return_value=["a", "b"]):
            stats = apply_overlap_separation(
                audio, sr, [OverlapRegion(0, 3.2, ["a", "b"])],
                assignments, profiles, object(), Path(temp), max_overlap_sec=1.4,
            )
        self.assertEqual([r["status"] for r in stats["regions"]], ["separated", "review", "separated"])
        self.assertEqual(stats["regions"][1]["reason"], "inference_failed")
        self.assertTrue(all(length <= 1400 for length in lengths))
        self.assertEqual(len(stats["overlap_writes"]["a"]), 2)

    def test_spilled_overlap_patches_keep_crossfades_and_render_identically(self):
        sr = 1000
        audio = np.ones(4300, dtype=np.float32)
        speakers = ("a", "b")
        assignments = [{"start": 0, "end": 4.3, "speaker_id": s} for s in speakers]
        profiles = {"a": np.array([1., 0.]), "b": np.array([0., 1.])}

        def separate(samples, sr, work_dir, *, backend):
            factor = 0.1 * (1 + int(work_dir.name.rsplit("_", 1)[1]))
            return samples * factor, samples * (1 - factor), "sepformer"

        with (
            TemporaryDirectory() as temp,
            patch("video2text.overlap_separation.separate_overlap_2spk", side_effect=separate),
            patch("video2text.overlap_separation._match_sources", return_value=list(speakers)),
            patch("video2text.speaker_stems._BLOCK_SAMPLES", 256),
        ):
            root = Path(temp)

            def render(name, threshold):
                directory = root / name
                directory.mkdir()
                with patch("video2text.overlap_separation._SPOOL_AUDIO_SECONDS", threshold):
                    result = apply_overlap_separation(
                        audio, sr, [OverlapRegion(0, 4.3, list(speakers))],
                        assignments, profiles, object(), directory, max_overlap_sec=1.4,
                    )
                writes = result.pop("overlap_writes")
                ranges = [(region["start_sample"], region["end_sample"]) for region in result["regions"]]
                tracks = {s: directory / f"{s}.wav" for s in speakers}
                stats = build_speaker_stems(
                    directory / "input.wav", assignments, tracks, sr=sr, audio=audio,
                    overlap_writes=writes, overlap_region_ranges=ranges, review_path=directory / "review.wav",
                )
                pcm = [sf.read(p, dtype="float32")[0].tobytes() for p in (*tracks.values(), directory / "review.wav")]
                return result, stats, pcm

            self.assertEqual(render("memory", 600), render("disk", 0))
            self.assertTrue(list((root / "disk" / "overlap_render").glob("*.npy")))

    def test_unresolved_overlap_is_preserved_at_original_offset(self):
        sr = 44100
        with TemporaryDirectory() as temp:
            path = Path(temp)
            vocals = path / "vocals.wav"
            audio = np.full(sr, 0.125, dtype=np.float32)
            sf.write(vocals, audio, sr, subtype="FLOAT")
            turns = [DiarizationTurn(0.125, 0.75, "a"), DiarizationTurn(0.5, 1, "b")]
            overlap = [OverlapRegion(0.5, 0.75, ["a", "b"])]
            for origin in ("timeline", "proximity"):
                diar = DiarizationResult(turns, overlap, "test", ["a", "b"], origin)
                with patch("video2text.speaker_stems.run_diarization", return_value=diar):
                    _, paths = split_vocals_by_speaker(
                        vocals, path, origin, path / "report.json",
                        split_mode="diarize_bss", use_embedding=False,
                    )

                review, out_sr = sf.read(path / f"{origin}_dialogue_review.wav")
                self.assertEqual(out_sr, sr)
                self.assertEqual(len(review), len(audio))
                np.testing.assert_array_equal(review[:sr // 8], 0.125)
                if origin == "timeline":
                    np.testing.assert_array_equal(review[sr // 2:3 * sr // 4], 0.125)
                else:
                    np.testing.assert_array_equal(review[sr // 8:], 0)
                tracks = [sf.read(track)[0] for track in paths.values()]
                np.testing.assert_array_equal(sum(tracks) + review, audio)
                report = json.loads((path / "report.json").read_text())
                self.assertEqual(report["overlap_bss"]["overlap_count"], 0)
                self.assertFalse(any(s["overlap_separated"] for s in report["segments"]))
                self.assertEqual(report["stats"]["coverage_ratio"], 0.875)

    def test_empty_diarization_retries_original_and_retains_unassigned_dialogue(self):
        sr = 44100
        with TemporaryDirectory() as temp:
            path = Path(temp)
            vocals = path / "vocals.wav"
            original = path / "mix.wav"
            audio = np.full(sr // 2, 0.0625, dtype=np.float32)
            sf.write(vocals, audio, sr, subtype="FLOAT")
            sf.write(original, audio, sr, subtype="FLOAT")
            empty = DiarizationResult([], [], "test")
            with patch("video2text.speaker_stems.run_diarization", return_value=empty) as run:
                _, paths = split_vocals_by_speaker(
                    vocals, path, "sample", path / "report.json",
                    split_mode="diarize_bss", original_audio_path=original,
                )
            self.assertEqual([call.args[0] for call in run.call_args_list], [vocals, original])
            self.assertEqual(paths, {})
            np.testing.assert_array_equal(sf.read(path / "sample_dialogue_review.wav")[0], audio)
            report = json.loads((path / "report.json").read_text())
            self.assertTrue(report["diarization_used_original_mix"])
            self.assertEqual(report["dialogue_review_reason"], "no_diarization_turns")


if __name__ == "__main__":
    unittest.main()
