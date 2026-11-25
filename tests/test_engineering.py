from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch
from zipfile import ZipFile

import numpy as np
import soundfile as sf

from video2text import bs_roformer_split, speaker_stems
from video2text.result_cache import render_cached_result
from video2text.stage_cache import StageCache, copy_bundle, inference_environment, publish_directory


class StageCacheTest(TestCase):
    def test_cuda_cache_lookup_can_resolve_identity_without_importing_torch(self):
        with (
            patch.dict(sys.modules, {"torch": None}),
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "2",
                                    "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}),
            patch("video2text.cuda_probe.wheel_cuda_identity", return_value=("gpu", "13.0")),
        ):
            self.assertEqual(inference_environment(fast=True), {
                "device": "gpu", "cuda": "13.0", "torch_threads": 2,
                "threads": {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"},
            })

    def test_complete_result_distinguishes_inputs_with_different_decode_paths(self):
        with (
            TemporaryDirectory() as folder,
            patch("video2text.result_cache.inference_environment", return_value={"device": "cpu"}),
            patch("video2text.result_cache.model_artifacts", return_value={}),
            patch("video2text.result_cache.ffprobe_bin", return_value=None),
        ):
            root = Path(folder)
            cache = StageCache(root / "cache")
            ffmpeg = root / "ffmpeg"
            ffmpeg.write_bytes(b"ffmpeg")
            wav, video = (root / "same.wav", root / "same.mka")
            for source in (wav, video):
                source.write_bytes(b"same input bytes")
            settings = {"roformer_model_dir": str(root)}
            calls = []

            def build(directory):
                calls.append(directory)
                (directory / "same_split_report.json").write_text(
                    json.dumps({"stats": {"vocals_duration_sec": 1.0}}),
                )

            for source, name in ((wav, "first"), (wav, "again"), (video, "decoded")):
                render_cached_result(cache, source, root / name, settings, ffmpeg, build)
            self.assertEqual(len(calls), 2)

    def test_roformer_validates_local_checkpoints_without_remote_requests(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            weights = root / "sample.ckpt"
            config = root / "sample.yaml"

            def write_checkpoint(path):
                with ZipFile(path, "w") as archive:
                    archive.writestr("archive/data.pkl", b"example")

            write_checkpoint(weights)
            config.write_text("config: example\n")
            separator = Mock()
            separator.list_supported_model_files.return_value = {
                "MDXC": {
                    "Sample": {"filename": weights.name, "download_files": [weights.name, config.name]},
                },
            }
            with (
                patch.object(bs_roformer_split, "_min_size_for", return_value=4),
                patch.object(bs_roformer_split, "_expected_remote_size", return_value=None) as remote,
            ):
                bs_roformer_split._ensure_model_artifacts_cached(separator, root, weights.name)
                remote.assert_not_called()
                separator.download_model_files.assert_not_called()
                # Keep a ZIP prefix but lose the central directory, as in a partial download.
                weights.write_bytes(weights.read_bytes()[:-22])
                separator.download_model_files.side_effect = lambda name: write_checkpoint(root / name)
                bs_roformer_split._ensure_model_artifacts_cached(separator, root, weights.name)
                separator.download_model_files.assert_called_once_with(weights.name)
                self.assertTrue(bs_roformer_split._is_artifact_complete(weights, None))

    def test_complete_bundle_is_reused_and_relocates_nested_preview(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            cache = StageCache(root / "cache")
            origins = []

            def build(directory):
                origins.append(directory)
                preview = directory / "overlap" / "preview.wav"
                preview.parent.mkdir()
                preview.write_bytes(b"separated")
                (directory / "demo_split_report.json").write_text(
                    json.dumps({"review": [{"preview": str(preview)}]}),
                )

            key = {"settings": "same"}
            source = cache.files("bundle", key, None, build)
            for name in ("first", "second"):
                hit = cache.files("bundle", key, None, build)
                self.assertEqual(hit, source)
                destination = root / name
                copy_bundle(hit, destination, origins[0])
                rewritten = json.loads((destination / "demo_split_report.json").read_text())
                preview = destination / "overlap" / "preview.wav"
                self.assertEqual(Path(rewritten["review"][0]["preview"]), preview)
                self.assertEqual(preview.read_bytes(), b"separated")
                self.assertFalse((destination / "complete.json").exists())
                preview.write_bytes(b"updated by user")
            self.assertEqual(len(origins), 1)
            self.assertEqual((source / "overlap" / "preview.wav").read_bytes(), b"separated")

    def test_failed_builder_is_retryable_and_updated_input_has_a_new_key(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            source.write_bytes(b"before")
            cache = StageCache(root / "cache")
            before = {"input": cache.fingerprint(source)}
            def fail(directory):
                (directory / "stage.wav").write_bytes(b"incomplete")
                raise RuntimeError("simulated termination")

            with self.assertRaises(RuntimeError):
                cache.files("test", before, ("stage.wav",), fail)
            built = []

            def make(directory):
                built.append(1)
                (directory / "stage.wav").write_bytes(b"complete")

            def resolve(_):
                return cache.files("test", before, ("stage.wav",), make)[0]

            with ThreadPoolExecutor(max_workers=3) as pool:
                hits = list(pool.map(resolve, range(3)))
            self.assertEqual(len(built), 1)
            self.assertEqual(hits, [hits[0]] * 3)
            self.assertEqual(hits[0].read_bytes(), b"complete")
            source.write_bytes(b"after")
            updated = {"input": cache.fingerprint(source)}
            (changed,) = cache.files("test", updated, ("stage.wav",), make)
            self.assertNotEqual(changed, hits[0])
            self.assertEqual(len(built), 2)

    def test_atomic_directory_rolls_back_and_rewrites_report_paths(self):
        with TemporaryDirectory() as folder:
            dest = Path(folder) / "中文"
            with self.assertRaises(RuntimeError):
                with publish_directory(dest, True) as staged:
                    (staged / "partial.wav").touch()
                    raise RuntimeError("failed")
            self.assertFalse(dest.exists())
            with publish_directory(dest, True) as staged:
                voice = staged / "spoken.wav"
                voice.touch()
                (staged / "demo_split_report.json").write_text(json.dumps({"vocals": str(voice)}))
            rewritten = json.loads((dest / "demo_split_report.json").read_text())
            self.assertEqual(Path(rewritten["vocals"]), dest / "spoken.wav")
            self.assertTrue((dest / "spoken.wav").is_file())
            with self.assertRaises(FileExistsError):
                with publish_directory(dest, True):
                    self.fail("existing output must not be replaced")


class StreamingRendererTest(TestCase):
    def test_seekable_stereo_matches_eager_loading_and_rendering(self):
        with TemporaryDirectory() as folder, patch.object(speaker_stems, "_BLOCK_SAMPLES", 64):
            path = Path(folder)
            sr = 44100
            src = path / "source.wav"
            data = np.random.default_rng(3).uniform(-1.0, 1.0, (534, 2))
            sf.write(src, data, sr, subtype="FLOAT")
            eager, _ = speaker_stems.load_vocals_mono(src, sr=sr)
            args = (src, [{"speaker_id": "a", "start": 0.0, "end": 480 / sr}])
            expected = speaker_stems.build_speaker_stems(
                *args, {"a": path / "expected.wav"}, sr=sr, audio=eager,
            )
            with patch.object(speaker_stems, "_SEEKABLE_AUDIO_SECONDS", 0):
                audio = speaker_stems._open_speaker_audio(src, sr)
                self.assertIsInstance(audio, speaker_stems._SeekableMono)
                np.testing.assert_array_equal(audio[-91:700], eager[-91:700])
                actual = speaker_stems.build_speaker_stems(*args, {"a": path / "actual.wav"}, sr=sr)
            self.assertEqual(actual, expected)
            self.assertEqual((path / "actual.wav").read_bytes(), (path / "expected.wav").read_bytes())

    def test_fill_spans_blocks_and_preserves_review_and_rounded_statistics(self):
        class Embedder:
            def embed_segment(self, samples, rate, start, end):
                return np.array([1.0, 0.0])

        with TemporaryDirectory() as folder, patch.object(speaker_stems, "_BLOCK_SAMPLES", 64):
            path = Path(folder)
            sr = 400
            signal = np.full(534, 0.25, dtype=np.float32)
            src = path / "source.wav"
            sf.write(src, signal, sr, subtype="FLOAT")
            track = path / "sp.wav"
            review = path / "review.wav"
            stats = speaker_stems.build_speaker_stems(
                src, [], {"a": track}, sr=sr, review_path=review, review_ranges=[(96, 192)],
                profiles={"a": np.array([1.0, 0.0])}, embedder=Embedder(), audio=signal,
            )
            self.assertEqual(stats["fill_unassigned_samples"], 438)
            self.assertEqual(stats["unassigned_vocal_sec"], 0.24)
            self.assertEqual(stats["review_sec"], 0.24)
            audio, _ = sf.read(track, dtype="float32")
            reserved, _ = sf.read(review, dtype="float32")
            np.testing.assert_array_equal(audio + reserved, signal)
            np.testing.assert_array_equal(audio[96:192], 0.0)
