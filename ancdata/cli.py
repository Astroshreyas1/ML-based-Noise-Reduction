"""Command-line entry point: `ancdata <command>` (installed) or `python -m ancdata.cli`.

Data commands        rir, registry, screen, import-mad, fixtures, snippets, pools
Generation           selftest, materialize, realcheck-mix
Battlefield v3       battlefield (build), battlefield-report, battlefield-listen, battlefield-selftest
Evaluation / plots   evaluate, plots, reality-gap
Recording session    mic-sweep, mic-fit, lombard-fit
Demo day             demo-check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="ancdata", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("rir", help="simulate the RIR bank (pyroomacoustics)")
    p.add_argument("--n-per-preset", type=int, default=60, help="rooms per preset (5 presets); 60 -> 300 rooms")
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("registry", help="index sources -> manifest.parquet (VAD screen on)")
    p.add_argument("--piles", nargs="*", help="subset of pile names; default = every pile present")
    p.add_argument("--no-screen", action="store_true", help="skip VAD (fixtures only)")

    p = sub.add_parser("screen", help="VAD-screen a noise pile; move flagged files to quarantine/")
    p.add_argument("--pile", required=True)
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("import-mad", help="copy a Military Audio Dataset checkout into the pile layout")
    p.add_argument("mad_root", type=Path)

    sub.add_parser("fixtures", help="write synthetic fixture sources into ANC_DATA_ROOT")

    p = sub.add_parser("snippets", help="materialise fixed-length labelled Lombard speech snippets (GRID + AVID)")
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--out", type=Path, help="default data/snippets/lombard<seconds>s")
    p.add_argument("--all-levels", action="store_true", help="also plain / normal / soft takes (default: Lombard-class only)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit-per-group", type=int, help="quick test: at most N snippets per (speaker, effort)")
    p.add_argument("--fmt", default="pcm16", choices=["float", "pcm16", "flac"])

    p = sub.add_parser("pools", help="index the raw noise corpora into split pools -> data/pools.parquet (voice screen on)")
    p.add_argument("--no-screen", action="store_true", help="skip the voice screen (MAD / FSD50K / UrbanSound8K)")
    p.add_argument("--workers", type=int, default=4)

    p = sub.add_parser("battlefield", help="build a split of the battlefield v3 dataset (memory-aware workers)")
    p.add_argument("--config", type=Path, default=Path("configs/battlefield.yaml"))
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--n", type=int, help="pairs; default = snippets x max_scenes_per_snippet[split]")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--out", type=Path)
    p.add_argument("--fmt", choices=["float", "pcm16", "flac"], help="default from config build.fmt[split]")
    p.add_argument("--workers", default="auto", help="int or 'auto' (RAM- and core-aware)")
    p.add_argument("--no-resume", action="store_true", help="regenerate ids already on disk")
    p.add_argument("--no-report", action="store_true")

    p = sub.add_parser("battlefield-report", help="composition tables + test baseline -> <root>/BUILD_REPORT.md")
    p.add_argument("--config", type=Path, default=Path("configs/battlefield.yaml"))
    p.add_argument("--root", type=Path, help="default data/<config name>")
    p.add_argument("--no-baseline", action="store_true")

    p = sub.add_parser("battlefield-listen", help="N pairs from the chain -> listening sheet (boom / ref / target)")
    p.add_argument("--config", type=Path, default=Path("configs/battlefield.yaml"))
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("outputs/battlefield_listen"))

    p = sub.add_parser("battlefield-selftest", help="contract, determinism, leakage, SNR bookkeeping, gunfire rate")
    p.add_argument("--config", type=Path, default=Path("configs/battlefield.yaml"))
    p.add_argument("--n", type=int, default=100)

    p = sub.add_parser("selftest", help="run the checks (or --smoke with no downloads)")
    p.add_argument("--config", type=Path)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--keep", type=Path)
    p.add_argument("--n", type=int, default=100)

    p = sub.add_parser("materialize", help="freeze N pairs to disk")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--fmt", default="float", choices=["float", "pcm16", "flac"],
                   help="float for frozen evals; flac for big training dumps (~4x smaller)")
    p.add_argument("--start", type=int, default=0, help="first example index (resume / shard)")
    p.add_argument("--stems", action="store_true", help="also write speech / per-layer noise / event stems")
    p.add_argument("--workers", type=int, default=1, help="processes; output identical to --workers 1")

    p = sub.add_parser("realcheck-mix", help="row 3a: real quiet-room speech x real venue noise -> frozen eval set")
    p.add_argument("--config", type=Path, default=Path("configs/eval_standard.yaml"))
    p.add_argument("--speech", type=Path, required=True, help="dir of quiet-room boom-mic takes")
    p.add_argument("--noise", type=Path, required=True, help="dir of venue/real noise recordings")
    p.add_argument("--impulsive", type=Path, help="dir of real gunshot files (optional)")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out", type=Path)

    p = sub.add_parser("evaluate", help="score a materialised set (baseline = unprocessed input)")
    p.add_argument("eval_root", type=Path)
    p.add_argument("--no-pesq", action="store_true")

    p = sub.add_parser("plots", help="crest/attack plot (synthetic vs real transients)")
    p.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    p.add_argument("--out", type=Path, default=Path("outputs/plots"))
    p.add_argument("--real-pile", default="esc50_gunshot")

    p = sub.add_parser("reality-gap", help="synthetic eval set vs real headset recordings")
    p.add_argument("--synthetic", type=Path, required=True, help="materialised eval dir")
    p.add_argument("--real", type=Path, required=True, help="dir of real headset recordings")
    p.add_argument("--out", type=Path, default=Path("outputs/plots/reality_gap.png"))

    p = sub.add_parser("mic-sweep", help="write the sine sweep to play at the mouth position")
    p.add_argument("--out", type=Path, default=Path("outputs/sweep.wav"))
    p.add_argument("--seconds", type=float, default=8.0)

    p = sub.add_parser("mic-fit", help="deconvolve the recorded sweep -> configs/mic_ir.npz")
    p.add_argument("--recorded", type=Path, required=True, help="boom-mic recording of the played sweep")
    p.add_argument("--seconds", type=float, default=8.0, help="same value used for mic-sweep")
    p.add_argument("--out", type=Path, default=Path("configs/mic_ir.npz"))

    p = sub.add_parser("lombard-fit", help="fit Lombard stats from calm/loud paired takes")
    p.add_argument("--pairs", nargs="+", required=True, help="calm1 loud1 calm2 loud2 ...")
    p.add_argument("--out", type=Path, default=Path("configs/lombard_stats.yaml"))

    p = sub.add_parser("demo-check", help="venue-day guard: is the live capture inside the training ranges?")
    p.add_argument("--ranges", type=Path, required=True, help="<eval set>/ranges.json")
    p.add_argument("--wav", type=Path, help="capture through the demo chain; omit to record 5 s (needs sounddevice)")

    a = ap.parse_args(argv)

    if a.cmd == "rir":
        from .rir_gen import generate_bank
        generate_bank(a.n_per_preset, a.seed)
    elif a.cmd == "registry":
        from .registry import build_manifest
        build_manifest(a.piles, screen=not a.no_screen)
    elif a.cmd == "screen":
        from .registry import screen_pile
        screen_pile(a.pile, move=not a.dry_run)
    elif a.cmd == "import-mad":
        from .registry import import_mad
        import_mad(a.mad_root)
    elif a.cmd == "fixtures":
        from .fixtures import make_fixtures
        make_fixtures()
    elif a.cmd == "snippets":
        from .paths import data_root
        from .snippets import ALL_LEVELS, LOMBARD_CLASS, build_snippets
        out = a.out or data_root() / "snippets" / f"lombard{a.seconds:g}s"
        build_snippets(out, a.seconds, ALL_LEVELS if a.all_levels else LOMBARD_CLASS, a.seed,
                       limit_per_group=a.limit_per_group, fmt=a.fmt)
        print(f"-> {out}")
    elif a.cmd == "pools":
        from .pools import build_pools
        build_pools(screen=not a.no_screen, workers=a.workers)
    elif a.cmd == "battlefield":
        from .battlefield import load_battlefield_config
        from .build import build_split, dataset_root, write_report
        cfg = load_battlefield_config(a.config)
        workers = a.workers if a.workers == "auto" else int(a.workers)
        build_split(cfg, a.split, a.n, a.out, workers=workers, fmt=a.fmt, resume=not a.no_resume, start=a.start)
        if not a.no_report and a.out is None:
            write_report(dataset_root(cfg), baseline=(a.split == "test"))
    elif a.cmd == "battlefield-report":
        from .battlefield import load_battlefield_config
        from .build import dataset_root, write_report
        cfg = load_battlefield_config(a.config)
        write_report(a.root or dataset_root(cfg), baseline=not a.no_baseline)
    elif a.cmd == "battlefield-listen":
        from .listen import listen_sheet
        print(listen_sheet(a.config, a.split, a.n, a.out, start=a.start))
    elif a.cmd == "battlefield-selftest":
        from .selftest import battlefield_selftest
        battlefield_selftest(a.config, a.n)
    elif a.cmd == "selftest":
        from .selftest import main as st
        args = ["--smoke"] if a.smoke else ["--config", str(a.config), "--n", str(a.n)]
        if a.keep:
            args += ["--keep", str(a.keep)]
        st(args)
    elif a.cmd == "materialize":
        from .materialize import materialize
        materialize(a.config, a.split, a.n, a.out, start_index=a.start, fmt=a.fmt, stems=a.stems, workers=a.workers)
    elif a.cmd == "realcheck-mix":
        from .config import load_config
        from .paths import eval_dir
        from .realcheck import mix_realcheck
        mix_realcheck(load_config(a.config), a.speech, a.noise, a.n, a.out or eval_dir("reality_a"), a.impulsive)
    elif a.cmd == "evaluate":
        from .metrics import evaluate_set, results_table
        df = evaluate_set(a.eval_root, with_pesq=not a.no_pesq)
        print(results_table(df))
        df.to_csv(a.eval_root / "baseline_metrics.csv", index=False)
        print(f"per-pair scores -> {a.eval_root / 'baseline_metrics.csv'}")
    elif a.cmd == "plots":
        from .config import load_config
        from .plots import plot_crest_attack
        print(plot_crest_attack(load_config(a.config), a.out / "crest_attack.png", real_pile=a.real_pile))
    elif a.cmd == "reality-gap":
        from .plots import plot_reality_gap
        files = sorted(p for p in a.real.rglob("*") if p.suffix.lower() in (".wav", ".flac"))
        print(plot_reality_gap(a.synthetic, files, a.out))
    elif a.cmd == "mic-sweep":
        from .audio import write_wav
        from .realcheck import make_sweep
        sweep, _ = make_sweep(a.seconds)
        write_wav(a.out, sweep)
        print(f"sweep -> {a.out}  (play at the mouth position, record on the boom mic, then `ancdata mic-fit`)")
    elif a.cmd == "mic-fit":
        from .audio import load_mono
        from .realcheck import fit_mic_ir, make_sweep, save_mic_ir
        _, inv = make_sweep(a.seconds)
        ir = fit_mic_ir(load_mono(a.recorded), inv)
        save_mic_ir(ir, a.out, note=f"from {a.recorded}")
        print(f"mic IR ({len(ir)} taps) -> {a.out}; train.yaml mic.ir_file already points here")
    elif a.cmd == "lombard-fit":
        from .lombard import fit_lombard_stats, save_stats
        if len(a.pairs) % 2:
            ap.error("--pairs needs calm/loud alternating")
        stats = fit_lombard_stats(list(zip(a.pairs[0::2], a.pairs[1::2])))
        save_stats(stats, a.out)
        print(stats, "->", a.out)
    elif a.cmd == "demo-check":
        from .demo_check import demo_capture_check
        st = demo_capture_check(a.ranges, a.wav)
        print("demo capture inside training ranges:", st)


if __name__ == "__main__":
    sys.exit(main())
