"""OpenFWI dataset split (seed 20261020), teacher density cache, and memory-mapped GPU waveform cache."""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import NamedTuple
import numpy as np
import torch
import wave_gauss_rs

DATASET_SEED = 20261020
N = 70
H = 10.0
CELLS = N * N
SHOTS = 5
NT = 2001
DT = 0.0005
PEAK_HZ = 15.0


class SplitEntry(NamedTuple):
    family: str
    file: str
    index: int
    hash_hex: str
    mean_v: int


class OpenFWIDatasetManager:
    """Builds and verifies the exact 20261020 OpenFWI split, teacher densities, and memmap wave cache."""

    def __init__(
        self,
        root_dir: Path | str | None = None,
        train_per_family: int = 400,
        val_per_family: int = 100,
        test_per_family: int = 100,
        seed: int = DATASET_SEED,
    ) -> None:
        if root_dir is None:
            root_dir = Path(__file__).resolve().parents[2]
        self.root_dir = Path(root_dir)
        self.openfwi_dir = self.root_dir / "data" / "openfwi"
        self.cache_dir = (
            self.root_dir
            / "data"
            / f"native_cache_s{seed}_{train_per_family}_{val_per_family}_{test_per_family}"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.train_per_family = train_per_family
        self.val_per_family = val_per_family
        self.test_per_family = test_per_family
        self.seed = seed

        with open(self.openfwi_dir / "manifest.json", "r", encoding="utf-8") as f:
            self.library = json.load(f)

        self.splits: dict[str, list[SplitEntry]] = {}
        self.fields_km: dict[str, np.ndarray] = {}
        self._build_or_load_splits()

    def _build_or_load_splits(self) -> None:
        manifest_path = self.cache_dir / "split_manifest.json"
        fields_path = self.cache_dir / "fields_km.npz"

        if manifest_path.exists() and fields_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for k in ("train", "validation", "test"):
                self.splits[k] = [SplitEntry(**item) for item in raw["splits"][k]]
            loaded = np.load(fields_path)
            for k in ("train", "validation", "test"):
                self.fields_km[k] = loaded[k]
            return

        splits: dict[str, list[SplitEntry]] = {"train": [], "validation": [], "test": []}
        arrays_mps: dict[str, list[np.ndarray]] = {"train": [], "validation": [], "test": []}
        seen_hashes: set[str] = set()

        for fi, fam in enumerate(self.library["families"]):
            train_entry = next(x for x in fam["files"] if x["split"] == "train")
            test_entry = next(x for x in fam["files"] if x["split"] == "test")

            train_arr = np.load(self.openfwi_dir / train_entry["path"], mmap_mode="r")
            test_arr = np.load(self.openfwi_dir / test_entry["path"], mmap_mode="r")

            train_models = train_arr.reshape(-1, N, N)
            test_models = test_arr.reshape(-1, N, N)

            needed_dev = self.train_per_family + self.val_per_family
            fam_seed = (self.seed ^ ((fi + 1) * 0x9E3779B9)) & 0xFFFFFFFF
            dev_indices = wave_gauss_rs.sample_unique_indices(
                len(train_models), needed_dev, fam_seed
            )
            test_indices = wave_gauss_rs.sample_unique_indices(
                len(test_models), self.test_per_family, (fam_seed ^ 0x85EBCA6B) & 0xFFFFFFFF
            )

            def record(
                split_key: str,
                models: np.ndarray,
                file_rel: str,
                indices: list[int],
            ) -> None:
                for idx in indices:
                    m = np.ascontiguousarray(models[int(idx)], dtype=np.float32)
                    flat = m.reshape(-1)
                    h_hex = wave_gauss_rs.hash_model_hex(flat)
                    if h_hex in seen_hashes:
                        raise ValueError(f"Duplicate geology hash {h_hex} across splits!")
                    seen_hashes.add(h_hex)
                    splits[split_key].append(
                        SplitEntry(
                            family=fam["id"],
                            file=file_rel,
                            index=int(idx),
                            hash_hex=h_hex,
                            mean_v=int(round(float(flat.mean()))),
                        )
                    )
                    arrays_mps[split_key].append(m)

            record("train", train_models, train_entry["path"], dev_indices[: self.train_per_family])
            record("validation", train_models, train_entry["path"], dev_indices[self.train_per_family : needed_dev])
            record("test", test_models, test_entry["path"], test_indices)

        self.splits = splits
        for k in ("train", "validation", "test"):
            self.fields_km[k] = np.stack(arrays_mps[k], axis=0).astype(np.float32) / 1000.0

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "schema": 1,
                    "seed": self.seed,
                    "perFamily": {
                        "train": self.train_per_family,
                        "validation": self.val_per_family,
                        "test": self.test_per_family,
                    },
                    "splits": {
                        k: [e._asdict() for e in v] for k, v in self.splits.items()
                    },
                },
                f,
                indent=2,
            )
        np.savez(fields_path, **self.fields_km)

    @property
    def train_mean_km(self) -> np.ndarray:
        """Return `[70, 70]` mean training velocity field in km/s."""
        return self.fields_km["train"].mean(axis=0).astype(np.float32)

    def get_validation_cohort_indices(self, per_family: int = 2) -> np.ndarray:
        """Select the first `per_family` validation examples per family (20 total by default)."""
        selected: list[int] = []
        n_fams = len(self.library["families"])
        for fi in range(n_fams):
            base = fi * self.val_per_family
            for k in range(min(per_family, self.val_per_family)):
                selected.append(base + k)
        return np.array(selected, dtype=np.int64)

    def get_reserved_test_cohort_indices(
        self, offset: int = 12, per_family: int = 2
    ) -> np.ndarray:
        """Select reserved test examples at `offset..offset+per_family` per family (matching `direct-gaussians.md`)."""
        selected: list[int] = []
        n_fams = len(self.library["families"])
        for fi in range(n_fams):
            base = fi * self.test_per_family
            eff_offset = offset if offset + per_family <= self.test_per_family else 0
            for k in range(min(per_family, self.test_per_family)):
                selected.append(base + eff_offset + k)
        return np.array(selected, dtype=np.int64)

    def get_train_teacher_densities(self) -> np.ndarray:
        """Compute or load `[N_train, 70, 70]` K=10 calibrated teacher densities via Rust Rayon."""
        path = self.cache_dir / "train_teacher_densities.npy"
        if path.exists():
            return np.load(path)
        train_flat = np.ascontiguousarray(
            self.fields_km["train"].reshape(len(self.fields_km["train"]), CELLS),
            dtype=np.float32,
        )
        densities = wave_gauss_rs.batch_teacher_densities(train_flat, N, N, 10)
        densities = densities.reshape(-1, N, N)
        np.save(path, densities)
        return densities

    def ensure_wave_cache(
        self,
        splits_to_simulate: tuple[str, ...] = ("train", "validation", "test"),
        engine: wave_gauss_rs.PyWaveEngine | None = None,
        on_progress: callable | None = None,
    ) -> dict[str, np.memmap]:
        """Simulate or open memory-mapped `[N_models, 5, 2001, 70]` waveform caches on disk."""
        if engine is None:
            engine = wave_gauss_rs.PyWaveEngine(SHOTS, NT, DT, PEAK_HZ, 20, 1000.0, 5000.0)

        # Also ensure background mean-velocity reference wavefield is cached
        bg_path = self.cache_dir / "mean_bg_traces.npy"
        if not bg_path.exists():
            mean_mps = np.ascontiguousarray(self.train_mean_km.reshape(-1) * 1000.0, dtype=np.float32)
            bg_traces = engine.simulate(mean_mps)
            np.save(bg_path, bg_traces)

        memmaps: dict[str, np.memmap] = {}
        for split in splits_to_simulate:
            models_km = self.fields_km[split]
            n_models = len(models_km)
            bin_path = self.cache_dir / f"waves_{split}.bin"
            done_path = self.cache_dir / f"waves_{split}.done.json"
            prog_path = self.cache_dir / f"waves_{split}.progress.json"
            shape = (n_models, SHOTS, NT, N)

            if bin_path.exists() and done_path.exists():
                memmaps[split] = np.memmap(bin_path, dtype=np.float32, mode="r", shape=shape)
                continue

            start_idx = 0
            sq_sum = 0.0
            if bin_path.exists() and prog_path.exists():
                try:
                    with open(prog_path, "r", encoding="utf-8") as f:
                        prog = json.load(f)
                    start_idx = int(prog.get("completed", 0))
                    sq_sum = float(prog.get("sq_sum", 0.0))
                except Exception:
                    start_idx = 0
                    sq_sum = 0.0

            mm_mode = "r+" if (start_idx > 0 and bin_path.exists()) else "w+"
            mm = np.memmap(bin_path, dtype=np.float32, mode=mm_mode, shape=shape)
            t0 = time.perf_counter()
            if start_idx > 0:
                print(f"[Wave Cache | {split}] Resuming native wgpu simulation from {start_idx}/{n_models}...")
            else:
                print(f"[Wave Cache | {split}] Simulating {n_models} 5-shot wavefields on native wgpu...")

            for i in range(start_idx, n_models):
                vel_mps = np.ascontiguousarray(models_km[i].reshape(-1) * 1000.0, dtype=np.float32)
                traces = engine.simulate(vel_mps)
                mm[i] = traces
                sq_sum += float(np.square(traces, dtype=np.float64).sum())
                done_count = i + 1
                if done_count % 100 == 0 or done_count == n_models:
                    mm.flush()
                    with open(prog_path, "w", encoding="utf-8") as f:
                        json.dump({"completed": done_count, "sq_sum": sq_sum}, f)
                    elapsed = max(1e-6, time.perf_counter() - t0)
                    rate = (done_count - start_idx) / elapsed
                    eta = (n_models - done_count) / max(rate, 1e-6)
                    if on_progress:
                        on_progress(split, done_count, n_models)
                    else:
                        print(
                            f"[Wave Cache | {split}] {done_count}/{n_models} "
                            f"({rate:.1f} models/s, ETA {eta:.0f}s)",
                            flush=True,
                        )
            mm.flush()
            rms = float(np.sqrt(sq_sum / (n_models * SHOTS * NT * N)))
            with open(done_path, "w", encoding="utf-8") as f:
                json.dump({"split": split, "count": n_models, "rms": rms}, f, indent=2)
            if prog_path.exists():
                prog_path.unlink()
            memmaps[split] = np.memmap(bin_path, dtype=np.float32, mode="r", shape=shape)

        return memmaps

    def get_normalization_stats(self) -> tuple[float, float, np.ndarray]:
        """Return `(global_rms_scale, scatter_rms_scale, mean_bg_traces [5, 2001, 70])` from training set only."""
        stats_path = self.cache_dir / "wave_norm_stats.json"
        bg_path = self.cache_dir / "mean_bg_traces.npy"
        done_path = self.cache_dir / "waves_train.done.json"
        if not bg_path.exists() or not done_path.exists():
            self.ensure_wave_cache(splits_to_simulate=("train",))

        bg_traces = np.load(bg_path)
        if stats_path.exists():
            with open(stats_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            return float(d["rms_scale"]), float(d["scatter_scale"]), bg_traces

        with open(done_path, "r", encoding="utf-8") as f:
            rms_scale = float(json.load(f)["rms"])

        train_mm = np.memmap(
            self.cache_dir / "waves_train.bin",
            dtype=np.float32,
            mode="r",
            shape=(len(self.fields_km["train"]), SHOTS, NT, N),
        )
        # Estimate scatter RMS over a deterministic stride of training examples
        step = max(1, len(train_mm) // 200)
        scatter_sq = 0.0
        count = 0
        for i in range(0, len(train_mm), step):
            diff = train_mm[i] - bg_traces
            scatter_sq += float(np.square(diff, dtype=np.float64).sum())
            count += diff.size
        scatter_scale = float(np.sqrt(scatter_sq / max(1, count)))

        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(
                {"rms_scale": rms_scale, "scatter_scale": max(scatter_scale, 1e-6)},
                f,
                indent=2,
            )
        return rms_scale, max(scatter_scale, 1e-6), bg_traces
