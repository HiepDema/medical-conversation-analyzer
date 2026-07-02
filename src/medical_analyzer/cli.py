"""CLI entry point for medical conversation analyzer."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import load_config
from .pipeline import MedicalPipeline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        "medical-analyzer",
        description="Analyze Vietnamese medical conversations: ASR + diarization + classification",
    )
    p.add_argument("audio_file", help="Path to audio file (.wav, .mp3, .flac)")
    p.add_argument("-o", "--output", help="Output JSON file (default: stdout)")
    p.add_argument("--no-correction", action="store_true", help="Skip LLM spell correction")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING - 10 * min(args.verbose, 2),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config()
    pipeline = MedicalPipeline(cfg)

    try:
        result = pipeline.analyze_file(args.audio_file)
    finally:
        pipeline.close()

    output = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Output saved to {args.output}", file=sys.stderr)
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
