from __future__ import annotations

import argparse
import json


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for image inference."""
    parser = argparse.ArgumentParser(description="Run ToothFairy inference and save selected visual outputs.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument(
        "--tasks",
        default="all",
        help="Comma-separated tasks: quadrants,teeth,periapical,teeth_classification,all",
    )
    parser.add_argument("--output-dir", default="outputs", help="Directory to write output images.")
    parser.add_argument("--conf-quadrants", type=float, default=0.3, help="Confidence threshold for quadrants.")
    parser.add_argument("--conf-teeth", type=float, default=0.3, help="Confidence threshold for teeth.")
    parser.add_argument(
        "--conf-periapical", type=float, default=0.3, help="Confidence threshold for periapical."
    )
    parser.add_argument(
        "--conf-teeth-classification",
        type=float,
        default=0.3,
        help="Confidence threshold for teeth classification (caries / impacted) on crops.",
    )
    parser.add_argument(
        "--no-quadrant-pathology-crops",
        action="store_true",
        help="Do not write per-quadrant JPEGs with periapical + classification overlays on each crop.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute inference pipeline and print generated artifact paths."""
    args = parse_args()
    from utils.pipeline import run_pipeline

    tasks = [task.strip() for task in args.tasks.split(",")]
    output = run_pipeline(
        image_path=args.image,
        tasks=tasks,
        output_dir=args.output_dir,
        conf_quadrants=args.conf_quadrants,
        conf_teeth=args.conf_teeth,
        conf_periapical=args.conf_periapical,
        conf_teeth_classification=args.conf_teeth_classification,
        save_quadrant_pathology_crops=not args.no_quadrant_pathology_crops,
    )
    printable = {key: str(path) for key, path in output.files.items()}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
