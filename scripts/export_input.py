#!/usr/bin/env python3
"""
export_input.py — Export EQBench3 scenarios to input.jsonl for batch inference.

Each scenario is flattened into a single prompt containing all turns + debrief,
separated by ===TURN=== delimiters. The model should respond to each turn in
sequence, separating responses with the same delimiter.

Usage:
    python scripts/export_input.py --output input.jsonl [--iterations 1]
"""

import argparse
import json
import logging
import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.benchmark import parse_scenario_prompts
from utils.constants import (
    STANDARD_SCENARIO_PROMPTS_FILE,
    STANDARD_MASTER_PROMPT_FILE,
    STANDARD_DEBRIEF_PROMPT_FILE,
    MESSAGE_DRAFTING_MASTER_PROMPT_FILE,
    ANALYSIS_MASTER_PROMPT_FILE,
    NO_RP_SCENARIO_IDS,
    MESSAGE_DRAFTING_SCENARIO_IDS,
    ANALYSIS_SCENARIO_IDS,
)

TURN_DELIMITER = "===TURN==="


def classify_scenario(scenario_id: str) -> str:
    """Return the scenario type string."""
    if scenario_id in ANALYSIS_SCENARIO_IDS:
        return "analysis"
    if scenario_id in MESSAGE_DRAFTING_SCENARIO_IDS:
        return "message_drafting"
    if scenario_id in NO_RP_SCENARIO_IDS:
        return "no_rp"
    return "standard"


def build_prompt(
    scenario_id: str,
    prompts: list[str],
    scenario_type: str,
    master_templates: dict[str, str],
    debrief_prompt: str,
) -> str:
    """Build a single flattened prompt for a scenario with all turns + debrief."""

    parts: list[str] = []
    has_debrief = scenario_type not in ("analysis",)
    num_turns = len(prompts)

    # Header instruction
    if has_debrief:
        total_sections = num_turns + 1  # turns + debrief
        parts.append(
            f"You will participate in a multi-turn scenario with {num_turns} "
            f"turn(s) followed by a debrief. Respond to each turn in sequence. "
            f"Separate each of your responses with the exact delimiter "
            f'"{TURN_DELIMITER}" on its own line.\n'
        )
    else:
        parts.append(
            f"You will respond to {num_turns} message(s) in sequence. "
            f"Separate each of your responses with the exact delimiter "
            f'"{TURN_DELIMITER}" on its own line.\n'
        )

    # Build each turn
    template = master_templates.get(scenario_type, "")
    for i, prompt_text in enumerate(prompts):
        turn_num = i + 1
        parts.append(f"=== TURN {turn_num} ===")

        # Apply master prompt template to every turn (except no_rp),
        # matching the original benchmark behavior in conversation.py
        if scenario_type != "no_rp" and template and "{scenario_prompt}" in template:
            formatted = template.format(scenario_prompt=prompt_text)
            parts.append(formatted)
        else:
            parts.append(prompt_text)

        parts.append("")  # blank line between turns

    # Debrief section
    if has_debrief:
        parts.append("=== DEBRIEF ===")
        parts.append(debrief_prompt)
        parts.append("")

    return "\n".join(parts).strip()


def main():
    parser = argparse.ArgumentParser(
        description="Export EQBench3 scenarios to input.jsonl for batch inference."
    )
    parser.add_argument(
        "--output", "-o",
        default="input.jsonl",
        help="Output JSONL file path (default: input.jsonl)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations per scenario (default: 1)",
    )
    parser.add_argument(
        "--verbosity",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.verbosity),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Load scenario prompts
    scenarios = parse_scenario_prompts(STANDARD_SCENARIO_PROMPTS_FILE)
    logging.info(f"Loaded {len(scenarios)} scenarios")

    # Load master prompt templates
    master_templates = {}
    for key, path in [
        ("standard", STANDARD_MASTER_PROMPT_FILE),
        ("message_drafting", MESSAGE_DRAFTING_MASTER_PROMPT_FILE),
        ("analysis", ANALYSIS_MASTER_PROMPT_FILE),
    ]:
        with open(path, "r", encoding="utf-8") as f:
            master_templates[key] = f.read()
        logging.info(f"Loaded master prompt template: {key} ({path})")

    # Load debrief prompt
    with open(STANDARD_DEBRIEF_PROMPT_FILE, "r", encoding="utf-8") as f:
        debrief_prompt = f.read().strip()
    logging.info(f"Loaded debrief prompt from {STANDARD_DEBRIEF_PROMPT_FILE}")

    # Generate input.jsonl
    count = 0
    with open(args.output, "w", encoding="utf-8") as out_f:
        for scenario_id in sorted(scenarios.keys(), key=lambda x: int(x)):
            prompts = scenarios[scenario_id]
            scenario_type = classify_scenario(scenario_id)

            for iteration in range(1, args.iterations + 1):
                prompt_text = build_prompt(
                    scenario_id=scenario_id,
                    prompts=prompts,
                    scenario_type=scenario_type,
                    master_templates=master_templates,
                    debrief_prompt=debrief_prompt,
                )

                has_debrief = scenario_type != "analysis"
                record = {
                    "id": f"scenario_{scenario_id}_iter_{iteration}",
                    "scenario_id": scenario_id,
                    "iteration": iteration,
                    "scenario_type": scenario_type,
                    "num_turns": len(prompts),
                    "has_debrief": has_debrief,
                    "prompt": prompt_text,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

    logging.info(f"Exported {count} items to {args.output}")
    print(f"Done. Exported {count} items to {args.output}")


if __name__ == "__main__":
    main()
