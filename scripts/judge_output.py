#!/usr/bin/env python3
"""
judge_output.py — Score model outputs using a judge model's rubric evaluation.

Reads input.jsonl (scenario metadata) and output.jsonl (model responses),
reconstructs conversation histories, and calls the judge model API for
rubric scoring.

Usage:
    python scripts/judge_output.py \
        --input input.jsonl \
        --output output.jsonl \
        --judge-model anthropic/claude-3.7-sonnet \
        --result-file results.json \
        --threads 4
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import statistics
import queue
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from core.benchmark import (
    parse_scenario_prompts,
    calculate_final_rubric_score,
    _execute_rubric_scoring_task,
)
from core.conversation import ScenarioTask
from utils.api import APIClient
from utils.file_io import load_json_file, save_json_file
import utils.constants as C

TURN_DELIMITER = "===TURN==="


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def split_model_output(raw_output: str, expected_parts: int) -> List[str]:
    """
    Split a model's raw output by the TURN delimiter into per-turn responses.
    Tries several fallback patterns if the primary delimiter is not found.
    """
    # Primary: exact delimiter
    if TURN_DELIMITER in raw_output:
        parts = raw_output.split(TURN_DELIMITER)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= expected_parts:
            return parts[:expected_parts]
        if parts:
            return parts

    # Fallback 1: ---TURN--- or similar
    for alt in [r"---\s*TURN\s*---", r"===\s*TURN\s*===", r"---\s*TURN\s+\d+\s*---"]:
        splits = re.split(alt, raw_output, flags=re.IGNORECASE)
        splits = [s.strip() for s in splits if s.strip()]
        if len(splits) >= expected_parts:
            return splits[:expected_parts]
        if len(splits) > 1:
            return splits

    # Fallback 2: === TURN N === or === DEBRIEF === headers
    header_pattern = r"===\s*(?:TURN\s+\d+|DEBRIEF)\s*==="
    splits = re.split(header_pattern, raw_output, flags=re.IGNORECASE)
    splits = [s.strip() for s in splits if s.strip()]
    if len(splits) >= expected_parts:
        return splits[:expected_parts]
    if len(splits) > 1:
        return splits

    # Last resort: return the whole output as a single part
    logging.warning(
        f"Could not split output into {expected_parts} parts. "
        f"Using entire output as single response."
    )
    return [raw_output.strip()]


def reconstruct_task(
    input_record: dict,
    output_text: str,
    scenario_prompts: Dict[str, List[str]],
    master_templates: Dict[str, str],
    debrief_prompt: str,
) -> Optional[ScenarioTask]:
    """
    Reconstruct a ScenarioTask from input metadata + model output text.
    """
    scenario_id = input_record["scenario_id"]
    iteration = input_record["iteration"]
    scenario_type = input_record["scenario_type"]
    num_turns = input_record["num_turns"]
    has_debrief = input_record["has_debrief"]

    prompts = scenario_prompts.get(scenario_id)
    if not prompts:
        logging.error(f"Scenario {scenario_id} not found in scenario_prompts.txt")
        return None

    # Determine expected number of output parts
    expected_parts = num_turns + (1 if has_debrief else 0)
    response_parts = split_model_output(output_text, expected_parts)

    # Separate turn responses and debrief
    if has_debrief and len(response_parts) >= num_turns + 1:
        turn_responses = response_parts[:num_turns]
        debrief_response = response_parts[num_turns]
    elif has_debrief and len(response_parts) == num_turns:
        # No debrief in output — use last turn as debrief fallback
        logging.warning(
            f"Scenario {scenario_id}: expected {expected_parts} parts but got "
            f"{len(response_parts)}. Missing debrief — using last response as debrief."
        )
        turn_responses = response_parts[:num_turns - 1] if num_turns > 1 else response_parts
        debrief_response = response_parts[-1]
    elif has_debrief:
        # Fewer parts than turns
        turn_responses = response_parts
        debrief_response = ""
        logging.warning(
            f"Scenario {scenario_id}: expected {expected_parts} parts but got "
            f"{len(response_parts)}. Debrief will be empty."
        )
    else:
        turn_responses = response_parts[:num_turns]
        debrief_response = None

    # Select the right master prompt template
    template = master_templates.get(scenario_type, "")

    # Build conversation history
    conversation_history: List[Dict[str, str]] = []
    parsed_responses: List[Dict[str, str]] = []

    for i, prompt_text in enumerate(prompts):
        # Format user message (same logic as ScenarioTask.run_scenario):
        # master template is applied to EVERY turn (except no_rp)
        if scenario_type != "no_rp" and template and "{scenario_prompt}" in template:
            formatted_prompt = template.format(scenario_prompt=prompt_text)
        else:
            formatted_prompt = prompt_text

        conversation_history.append({"role": "user", "content": formatted_prompt})

        # Get model response for this turn
        if i < len(turn_responses):
            response = turn_responses[i]
        else:
            response = ""
            logging.warning(
                f"Scenario {scenario_id}: no response for turn {i + 1}. "
                f"Using empty string."
            )

        conversation_history.append({"role": "assistant", "content": response})

        # Parse structured sections for standard/drafting; raw for others
        is_no_rp = scenario_id in C.NO_RP_SCENARIO_IDS
        is_analysis = scenario_id in C.ANALYSIS_SCENARIO_IDS
        if not is_no_rp and not is_analysis:
            # Create a temporary ScenarioTask just for parsing
            tmp = ScenarioTask(
                scenario_id=scenario_id,
                prompts=prompts,
                debrief_prompt=None,
                iteration_index=iteration,
                test_model="batch-model",
            )
            parsed_entry = tmp._parse_response(response)
        else:
            parsed_entry = {"raw": response}

        parsed_responses.append(parsed_entry)

    # Create the ScenarioTask object
    task = ScenarioTask(
        scenario_id=scenario_id,
        prompts=prompts,
        debrief_prompt=debrief_prompt if has_debrief else None,
        iteration_index=iteration,
        test_model="batch-model",
        master_prompt_template=template if scenario_type != "no_rp" else None,
    )
    task.conversation_history = conversation_history
    task.parsed_responses = parsed_responses
    task.debrief_response = debrief_response
    task.start_time = time.time()

    # Set status based on whether debrief is expected
    if has_debrief:
        task.status = "completed"
    else:
        task.status = "scenario_completed"

    return task


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score model outputs using judge model rubric evaluation."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input.jsonl (generated by export_input.py)",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output.jsonl (model responses from internal system)",
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("JUDGE_MODEL_NAME"),
        help="Judge model API identifier (default: JUDGE_MODEL_NAME from .env)",
    )
    parser.add_argument(
        "--model-name",
        default="batch-model",
        help="Logical name for the tested model (default: batch-model)",
    )
    parser.add_argument(
        "--result-file",
        default="eqbench3_runs.json",
        help="Output result file (default: eqbench3_runs.json)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of parallel threads for judge API calls (default: 4)",
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

    if not args.judge_model:
        logging.error(
            "No judge model specified. Use --judge-model or set "
            "JUDGE_MODEL_NAME in .env"
        )
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 1. Load input.jsonl and output.jsonl, match by id
    # -----------------------------------------------------------------------
    logging.info(f"Reading input from {args.input}")
    input_records: Dict[str, dict] = {}
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            input_records[rec["id"]] = rec

    logging.info(f"Reading output from {args.output}")
    output_records: Dict[str, str] = {}
    with open(args.output, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            record_id = rec.get("id")
            output_text = rec.get("output", "")
            if record_id:
                output_records[record_id] = output_text
            else:
                logging.warning("Output record missing 'id' field, skipping.")

    # Match
    matched_ids = set(input_records.keys()) & set(output_records.keys())
    if not matched_ids:
        logging.error("No matching IDs between input and output files!")
        sys.exit(1)
    logging.info(
        f"Matched {len(matched_ids)} / {len(input_records)} scenarios"
    )

    # -----------------------------------------------------------------------
    # 2. Load scenario data and templates
    # -----------------------------------------------------------------------
    scenario_prompts = parse_scenario_prompts(C.STANDARD_SCENARIO_PROMPTS_FILE)
    logging.info(f"Loaded {len(scenario_prompts)} scenarios from prompts file")

    master_templates = {}
    for key, path in [
        ("standard", C.STANDARD_MASTER_PROMPT_FILE),
        ("message_drafting", C.MESSAGE_DRAFTING_MASTER_PROMPT_FILE),
        ("analysis", C.ANALYSIS_MASTER_PROMPT_FILE),
    ]:
        master_templates[key] = Path(path).read_text(encoding="utf-8")

    debrief_prompt = Path(C.STANDARD_DEBRIEF_PROMPT_FILE).read_text(
        encoding="utf-8"
    ).strip()

    # -----------------------------------------------------------------------
    # 3. Load rubric scoring templates
    # -----------------------------------------------------------------------
    # Standard rubric
    with open(C.STANDARD_RUBRIC_CRITERIA_FILE, "r", encoding="utf-8") as f:
        standard_criteria = [
            l.strip() for l in f if l.strip() and not l.strip().startswith("#")
        ]
    output_fmt_std = {
        "chain_of_thought_reasoning":
            "detailed chain of thought reasoning about the coming scoring decisions"
    }
    for c in standard_criteria:
        output_fmt_std[c] = 0
    standard_rubric_output_format = json.dumps(
        output_fmt_std, indent=2
    ).replace(": 0", ": 0-20")
    standard_rubric_template = Path(
        C.STANDARD_RUBRIC_PROMPT_FILE
    ).read_text(encoding="utf-8")

    # Analysis rubric
    with open(C.ANALYSIS_RUBRIC_CRITERIA_FILE, "r", encoding="utf-8") as f:
        analysis_criteria = [
            l.strip() for l in f if l.strip() and not l.strip().startswith("#")
        ]
    output_fmt_anl = {
        "chain_of_thought_reasoning":
            "detailed chain of thought reasoning about the coming scoring decisions"
    }
    for c in analysis_criteria:
        output_fmt_anl[c] = 0
    analysis_rubric_output_format = json.dumps(
        output_fmt_anl, indent=2
    ).replace(": 0", ": 0-20")
    analysis_rubric_template = Path(
        C.ANALYSIS_RUBRIC_PROMPT_FILE
    ).read_text(encoding="utf-8")

    logging.info(
        f"Loaded rubric templates: {len(standard_criteria)} standard criteria, "
        f"{len(analysis_criteria)} analysis criteria"
    )

    # -----------------------------------------------------------------------
    # 4. Reconstruct ScenarioTask objects
    # -----------------------------------------------------------------------
    tasks: List[ScenarioTask] = []
    for record_id in sorted(matched_ids):
        input_rec = input_records[record_id]
        output_text = output_records[record_id]

        task = reconstruct_task(
            input_rec, output_text, scenario_prompts,
            master_templates, debrief_prompt,
        )
        if task:
            task.model_name = args.model_name
            tasks.append(task)

    logging.info(f"Reconstructed {len(tasks)} tasks for rubric scoring")

    if not tasks:
        logging.error("No tasks to score!")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 5. Run rubric scoring via judge API
    # -----------------------------------------------------------------------
    judge_api = APIClient(model_type="judge")
    api_clients = {"judge": judge_api}

    # Create a simple save queue (no-op save worker — we save at the end)
    save_q = queue.Queue()
    run_key = f"batch_{args.model_name}"

    logging.info(
        f"Starting rubric scoring with judge model: {args.judge_model} "
        f"({args.threads} threads)"
    )

    scored_tasks = []
    errors = 0

    with tqdm(total=len(tasks), desc="Rubric scoring") as pbar:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = {}
            for task in tasks:
                is_analysis = task.scenario_id in C.ANALYSIS_SCENARIO_IDS
                rubric_template = (
                    analysis_rubric_template
                    if is_analysis
                    else standard_rubric_template
                )
                rubric_format = (
                    analysis_rubric_output_format
                    if is_analysis
                    else standard_rubric_output_format
                )

                future = executor.submit(
                    _execute_rubric_scoring_task,
                    task=task,
                    api_clients=api_clients,
                    judge_model_id=args.judge_model,
                    rubric_prompt_template=rubric_template,
                    rubric_output_format_str=rubric_format,
                    save_queue=save_q,
                    run_key=run_key,
                    truncate_for_rubric=False,
                )
                futures[future] = task

            for future in as_completed(futures):
                task = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logging.error(
                        f"Scoring failed for scenario {task.scenario_id} "
                        f"(iter {task.iteration_index}): {e}"
                    )

                if task.status == "rubric_scored":
                    scored_tasks.append(task)
                else:
                    errors += 1
                    logging.warning(
                        f"Task {task.scenario_id} (iter {task.iteration_index}) "
                        f"ended with status: {task.status}"
                    )
                pbar.update(1)

    # Drain the save queue (discard — we build run_data ourselves)
    while not save_q.empty():
        try:
            save_q.get_nowait()
        except queue.Empty:
            break

    logging.info(
        f"Scoring complete: {len(scored_tasks)} scored, {errors} errors"
    )

    # -----------------------------------------------------------------------
    # 6. Calculate and display results
    # -----------------------------------------------------------------------
    # Build run_data structure compatible with calculate_final_rubric_score
    run_data = {
        "run_key": run_key,
        "model_name": args.model_name,
        "judge_model": args.judge_model,
        "status": "completed",
        "start_time": min(
            (t.start_time for t in tasks if t.start_time), default=None
        ),
        "end_time": time.time(),
        "scenario_tasks": {},
    }

    for task in tasks:
        iter_str = str(task.iteration_index)
        if iter_str not in run_data["scenario_tasks"]:
            run_data["scenario_tasks"][iter_str] = {}
        run_data["scenario_tasks"][iter_str][task.scenario_id] = task.to_dict()

    avg_score, error_msg = calculate_final_rubric_score(run_data)

    if avg_score is not None:
        scaled_score = round(avg_score * 5, 2)  # 0-20 → 0-100
        run_data["results"] = {
            "average_rubric_score": avg_score,
            "rubric_score_0_100": scaled_score,
        }

        print()
        print("=" * 60)
        print("        EQBench3 Rubric Score Summary")
        print("=" * 60)
        print(f"  Model:          {args.model_name}")
        print(f"  Judge:          {args.judge_model}")
        print(f"  Tasks scored:   {len(scored_tasks)} / {len(tasks)}")
        print(f"  Rubric (0-20):  {avg_score:.2f}")
        print(f"  Rubric (0-100): {scaled_score:.2f}")
        print("=" * 60)
        print()
    else:
        print(f"\nFailed to calculate rubric score: {error_msg}\n")

    # Save results
    result_data = load_json_file(args.result_file) or {}
    result_data[run_key] = run_data
    save_json_file(args.result_file, result_data)
    logging.info(f"Results saved to {args.result_file}")
    print(f"Results saved to {args.result_file}")


if __name__ == "__main__":
    main()
