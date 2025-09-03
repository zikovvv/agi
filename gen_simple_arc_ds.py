from enum import Enum
from augmentation_utils import transform_shuffle_colors
from get_augmented_ds import *
from re_arc.main import *

gens = get_generators()
vers = get_verifiers()

    
class PuzzleNames(Enum) :
    FILL_SIMPLE_OPENED_SHAPE = '444801d8'
    DENOISING = '5614dbcf'
    MOVE_RANDOM_NOISE_INTO_OUTLINED_BY_CORNERS_PLACE = 'a1570a43'
    FILL_NOISED_CLOSED_SHAPES_CONNECTED = '00d62c1b'


    DRAW_ZIGZAG_PATTERN_FROM_POINTS_IN_THE_CORNERS_OF_FIELDS = 'e179c5f4'
    REPEAT_SHAPE_BASED_ON_SOME_ADJACENT_COLORED_CELLS_THAT_MARK_DIRECTION = 'e8dc4411'


def gen_arc_puzzle_ex(
    name : PuzzleNames | str,
    nb_examples = 100,
    augment_colors : bool = False,
    max_colors_augmentations : int = 10,
    do_shuffle : bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
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
    name = PuzzleNames.FILL_SIMPLE_OPENED_SHAPE
    name = 'e8dc4411'
    examples = gen_arc_puzzle_ex(name)
    examples = examples[:10] 
    show_examples(examples)
    
if __name__ == "__main__":
    main()
    