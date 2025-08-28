from get_augmented_ds import *
from re_arc.main import *

gens = get_generators()
vers = get_verifiers()

def gen_examples(name : str, nb_examples = 100) :
    g = gens[name]
    examples = []
    for _ in range(nb_examples) :
        ex = g(0, 1)
        # print(ex)
        q, a = np.array(ex['input']), np.array(ex['output'])
        qa = (q, a)
        examples.append(qa)

    return examples

def gen_simple_ds_denoising_1(nb_examples = 1000) :
    name = '5614dbcf'
    examples = gen_examples(name, nb_examples=nb_examples)
    return examples

def gen_simple_ds_fill_simple_shape(nb_examples = 1000) :
    name = '444801d8'
    examples = gen_examples(name, nb_examples=nb_examples)
    return examples

def main() :
    name = '444801d8'
    examples = gen_examples(name)
    examples = examples[:10] 
    show_example(examples)
    
if __name__ == "__main__":
    main()
    