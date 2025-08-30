from enum import Enum
from augmentation_utils import transform_shuffle_colors
from get_augmented_ds import *
from re_arc.main import *

gens = get_generators()
vers = get_verifiers()

    
class PuzzleNames(Enum) :
    FIll_SIMPLE_OPENED_SHAPE = '444801d8'
    DENOISING = '5614dbcf'


def gen_examples(
    name : PuzzleNames | str,
    nb_examples = 100,
    augment_colors : bool = False,
    max_colors_augmentations : int = 10,
    do_shuffle : bool = True,
) :
    if isinstance(name, PuzzleNames):
        name = name.value

    g = gens[name]
    examples = []
    for _ in range(nb_examples) :
        ex = g(0, 1)
        # print(ex)
        q, a = np.array(ex['input']), np.array(ex['output'])
        qa = (q, a)
        if augment_colors :
            augmented_colors = transform_shuffle_colors(q, a)[:max_colors_augmentations]
            examples.extend(augmented_colors)
        else : 
            examples.append(qa)
            
    if do_shuffle:
        np.random.shuffle(examples)
    examples = examples[:nb_examples]
    return examples

def main() :
    name = PuzzleNames.FIll_SIMPLE_OPENED_SHAPE
    examples = gen_examples(name)
    examples = examples[:10] 
    show_example(examples)
    
if __name__ == "__main__":
    main()
    