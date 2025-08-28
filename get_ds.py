import json
import os
from typing import Dict, Tuple, List
import numpy as np

DIR_KAGGLE_2025 = './data/arc_challenge_2025'
EXAMPLES_FILE_KAGGLE_2025 = os.path.join(DIR_KAGGLE_2025, 'arc-agi_training_challenges.json')
SOLUTIONS_FILE_KAGGLE_2025 = os.path.join(DIR_KAGGLE_2025, 'arc-agi_training_solutions.json')

DIR_KAGGLE_2024 = './data/arc-prize-2024'
EXAMPLES_FILE_KAGGLE_2024 = os.path.join(DIR_KAGGLE_2024, 'arc-agi_training_challenges.json')
SOLUTIONS_FILE_KAGGLE_2024 = os.path.join(DIR_KAGGLE_2024, 'arc-agi_training_solutions.json')


def get_ds() -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    ex_kaggle_2025 = json.load(open(EXAMPLES_FILE_KAGGLE_2025, 'r'))
    solutions_kaggle_2025 = json.load(open(SOLUTIONS_FILE_KAGGLE_2025, 'r'))

    ex_kaggle_2024 = json.load(open(EXAMPLES_FILE_KAGGLE_2024, 'r'))
    solutions_kaggle_2024 = json.load(open(SOLUTIONS_FILE_KAGGLE_2024, 'r'))

    examples = {**ex_kaggle_2025, **ex_kaggle_2024}
    solutions = {**solutions_kaggle_2025, **solutions_kaggle_2024}
    dataset : Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    for ex_id in examples.keys() :
        example = examples[ex_id]
        solution = solutions[ex_id]
        res = []
        # print(example)
        for a in example['train'] :
            ex, sol = a['input'], a['output']
            res.append((np.array(ex), np.array(sol)))

        res.append((np.array(example['test'][0]['input']), np.array(solution[0])))
        # print(res[-1])
        # exit()
        dataset[ex_id] = res
    return dataset


