import json
import os
from typing import Dict, Tuple, List

data_train_files = os.listdir('./data/training')

dataset_train : List[Dict] = [json.load(open(os.path.join('./data/training', file), 'r')) for file in data_train_files]
    

import matplotlib.pyplot as plt

# show as grid List[List[int]] this showas grid

def show_grid(example : List[List[int]], title : str = ''):
    plt.imshow(example, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()



def show_example(example : Dict, title : str = ''):
    train :List[Tuple[List[List[int]], List[List[int]]]] = [(item['input'], item['output']) for item in example['train']]
    test :List[Tuple[List[List[int]], List[List[int]]]] = [(item['input'], item['output']) for item in example['test']]
    for i, (train_input, train_output) in enumerate(train):
        show_grid(train_input, title=f'{title} - Train Input {i+1}')
        show_grid(train_output, title=f'{title} - Train Output {i+1}')

    for i, (test_input, test_output) in enumerate(test):
        show_grid(test_input, title=f'{title} - Test Input {i+1}')
        show_grid(test_output, title=f'{title} - Test Output {i+1}')
        
        
def main():
    for example in dataset_train:
        show_example(example)
        exit()

if __name__ == "__main__":
    main()