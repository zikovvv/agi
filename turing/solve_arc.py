from show import *

idx = 1
show_examples(list(dataset.values())[idx])



def solve(task : np.ndarray, sol : np.ndarray) :
    task_w, task_h = task.shape
    sol_w, sol_h = sol.shape
    # sol is zeros
    if np.count_nonzero(sol) == 0:
        return task

    