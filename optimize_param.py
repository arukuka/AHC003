#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from pathlib import Path
import json
import optuna
import tempfile
import subprocess


def run(target_path, tester_path, inputs_dir, param_path, index):
    input_file_path = inputs_dir / Path(f'{index:04}.txt')
    proc = subprocess.run(
        f"{tester_path} {input_file_path} {target_path} -p {param_path}",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE)
    stderr = proc.stderr.decode()
    stderr = [line for line in stderr.splitlines() if 'Score = ' in line][0]
    score = int(stderr.split()[-1])

    return score


def objective(trial, target_path, tester_path, inputs_dir):
    param_list = [
        "width_ratio",
        "width_decrease_ratio",
    ]
    config = dict()
    for param_name in param_list:
        config[param_name] = trial.suggest_uniform(param_name, 0.0, 1.0 + 1e-6)
    config["use_analyze_thr"] = trial.suggest_int("use_analyze_thr", 0, 10)
    config["default_value"] = trial.suggest_int("default_value", 1000, 9000)
    config["default_width"] = trial.suggest_int("defalt_width", 1000, 9000)

    NUM_TEST = 100

    with tempfile.TemporaryDirectory() as dname:
        temp_dir = Path(dname)
        param_path = temp_dir / Path("param.json")
        with open(param_path, mode='w') as f:
            f.write(json.dumps(config))

        score = 0
        for index in range(NUM_TEST):
            raw_score = run(target_path, tester_path, inputs_dir, param_path, index)
            score = score - raw_score

            trial.report(score, index)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return score


def main():
    parser = ArgumentParser()
    parser.add_argument("target", type=Path, help="target binary file")
    parser.add_argument("tester", type=Path, help="AHC003 tester file")
    parser.add_argument("inputs_dir", type=Path, help="AHC003 in dir")
    parser.add_argument("--study")
    parser.add_argument("--storage")
    parser.add_argument("--show", action='store_true')

    args = parser.parse_args()
    assert args.target.is_file()
    assert args.tester.is_file()
    assert args.inputs_dir.is_dir()

    if args.study and args.storage:
        print("LOAD STUDY")
        study = optuna.load_study(study_name=args.study, storage=args.storage)
    else:
        print("CREATE STUDY")
        study = optuna.create_study(pruner=optuna.pruners.MedianPruner())

    if args.show:
        print("Best params: ", study.best_params)
        print("Best value: ", study.best_value)
        return

    study.optimize(
        lambda trial: objective(
            trial,
            args.target.resolve(),
            args.tester.resolve(),
            args.inputs_dir.resolve()),
        n_jobs=-1)


if __name__ == "__main__":
    main()
