from concurrent.futures import ProcessPoolExecutor
import itertools

import tqdm
from augmentation_utils import augment_example_geometry_and_colors
from get_ds import *
from show import *



def ex1 ():
    dataset = get_ds()
    example_task = """
    0 1 2 3 
    0 1 2 0
    0 1 2 0
    """
    example_task = np.array([[int(x) for x in line.split()] for line in example_task.strip().split('\n')])
    example_answer  = """
    0 1 2 0
    0 1 2 0
    0 1 2 0
    """


    example_answer = np.array([[int(x) for x in line.split()] for line in example_answer.strip().split('\n')])

    all_augmentations = augment_example_geometry_and_colors(example_task, example_answer)
    print(len(all_augmentations))

    # all_augmentations_unique = [(np.array(t_flat).reshape(t_shape), np.array(a_flat).reshape(a_shape)) for t_shape, a_shape, t_flat, a_flat in list({(t.shape, a.shape, tuple(t.flatten()), tuple(a.flatten())) for t, a in all_augmentations})]
    # print(len(all_augmentations_unique))
    
    

    show_examples(all_augmentations[:15])
    
    augmented_ds : List[List[List[Tuple[np.ndarray, np.ndarray]]]] = [
        [augment_example_geometry_and_colors(t, a) for t, a in dataset[ex_id]]
        for ex_id in tqdm.tqdm(list(dataset.keys())[:10])
    ]

    augmented_ds_serializable : List[List[List[Tuple[List[List[int]], List[List[int]]]]]] = [
        [[(tt.tolist(), aa.tolist()) for tt, aa in example] for example in task] for task in augmented_ds
    ]

    j = 0
    print(len(augmented_ds))
    print(len(augmented_ds[j]))
    print(len(augmented_ds[j][0]))
    show_examples(augmented_ds[j][0][:20])


def get_full_ds() :
    augmented_ds : List[List[List[Tuple[np.ndarray, np.ndarray]]]] = [
        [augment_example_geometry_and_colors(t, a) for t, a in dataset[ex_id]]
        for ex_id in tqdm.tqdm(list(dataset.keys()))
    ]

    augmented_ds_serializable : List[List[List[Tuple[List[List[int]], List[List[int]]]]]] = [
        [[(tt.tolist(), aa.tolist()) for tt, aa in example] for example in task] for task in augmented_ds
    ]

    print(len(augmented_ds))
    print(len(augmented_ds[0]))
    print(len(augmented_ds[1]))

    # save as json
    if 1 :
        with open("./data/augmented_dataset.json", "w") as f:
            json.dump(augmented_ds_serializable, f)
    else :
        nb_examples_in_subset = 10
        with open("./data/augmented_dataset_subset.json", "w") as f:
            json.dump(augmented_ds_serializable, f)
        
    
    show_examples(augmented_ds[3][0][:15])
    
if __name__ == "__main__":
    if 1 :
        get_full_ds()
    else :
        ex1()
