
# def gen_task_outline_object(
#     w : int = 15,
#     h : int = 15,
#     obj_size : int = 25,
#     outline_size : int = 1
# ) -> Tuple[np.ndarray, np.ndarray]:
#     question, answer = np.zeros((h, w)), np.zeros((h, w))

#     object_start_point_coords = np.random.randint(
#         outline_size, min(h, outline_size + obj_size), size=2
#     )

#     cur_coord = 
#     for i in range(obj_size):
#         question[object_start_point_coords[0] + i, object_start_point_coords[1]: object_start_point_coords[1] + obj_size] = 1
