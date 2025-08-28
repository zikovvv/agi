from concurrent.futures import ProcessPoolExecutor
import itertools
import tqdm
from show import *

# COLORS = [_ for _ in range(10)]
all_permutations = [
    list(itertools.permutations([_ for _ in range(nb)], nb)) for nb in range(1, 11)
]
# print(all_permutations[1])
# exit()
for perms in all_permutations : 
    print(len(perms))
# exit()


MAX_NB_PERMUTATIONS = 15
def transform_shuffle_colors(task : np.ndarray, answer : np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]] :
    colors = np.unique(task)
    colors1 = np.unique(answer)
    colors = np.union1d(colors, colors1)
    nb_colors = len(colors)
    res = []
    task_masks = [(task == c) for c in colors]
    answer_masks = [(answer == c) for c in colors]
    perms = all_permutations[nb_colors - 1]
    if len(perms) > MAX_NB_PERMUTATIONS:
        # rng = np.random.default_rng()
        perms = random.sample(perms, k=MAX_NB_PERMUTATIONS)
    for perm in perms:
        task_perm  = np.zeros_like(task)
        answer_perm = np.zeros_like(answer)
        for j, diff in enumerate(task_masks):
            task_perm += diff * perm[j]
        for j, diff in enumerate(answer_masks):
            answer_perm += diff * perm[j]
        res.append((task_perm, answer_perm))
    return res



def transform_mirror(task: np.ndarray, answer: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    res = []
    task_mirror_h = np.flip(task, axis=1)
    answer_mirror_h = np.flip(answer, axis=1)

    res.append((task_mirror_h, answer_mirror_h))
    task_mirror_v = np.flip(task, axis=0)
    answer_mirror_v = np.flip(answer, axis=0)
    res.append((task_mirror_v, answer_mirror_v))

    return res

def transform_transpose(task: np.ndarray, answer: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    res = []
    task_transpose_main_diagonal = np.transpose(task)
    answer_transpose_main_diagonal = np.transpose(answer)
    res.append((task_transpose_main_diagonal, answer_transpose_main_diagonal))

    task_transpose_anti_diagonal = np.flip(np.transpose(task), axis=1)
    answer_transpose_anti_diagonal = np.flip(np.transpose(answer), axis=1)
    res.append((task_transpose_anti_diagonal, answer_transpose_anti_diagonal))

    return res

def transform_rotate(task: np.ndarray, answer: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    res = []
    for _ in range(4):
        task = np.rot90(task)
        answer = np.rot90(answer)
        res.append((task, answer))
    return res

def augment_example_geometry(task: np.ndarray, answer: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    augmented = []
    augmented.extend(transform_mirror(task, answer))
    augmented.extend(transform_transpose(task, answer))
    augmented.extend(transform_rotate(task, answer))
    return augmented

def augment_example_geometry_and_colors(task: np.ndarray, answer: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    augmented = augment_example_geometry(task, answer)
    augmented.extend([b for t, a in augmented for b in transform_shuffle_colors(t, a)])
    augmented_unique = [(np.array(t_flat).reshape(t_shape), np.array(a_flat).reshape(a_shape)) for t_shape, a_shape, t_flat, a_flat in list({(t.shape, a.shape, tuple(t.flatten()), tuple(a.flatten())) for t, a in augmented})]
    return augmented_unique
