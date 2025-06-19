import pickle

from mpi4py import MPI
from schwarz import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def serialize_objects(obj_list):
    return pickle.dumps(obj_list)


def deserialize_objects(data):
    return pickle.loads(data)


if __name__ == '__main__':
    # initial_rects: list[Rectangle] = get_diagonal_rects(10, 4)  # get_two_rects()
    # initial_sectors: list[PolarSector] = [PolarSector(40, 40, 10, 20, 0, np.pi * 2, 0.5, 0.1).set_boundary(boundary)]
    rects_sectors: list[Rectangle | PolarSector] = get_my_version(divisions=size)
    local_A = [rects_sectors[r] for r in range(0, len(rects_sectors)) if r % 2 == 0][rank::size]
    print(f"[Rank {rank}] Размер локальной подгруппы A: {len(local_A)}, состав: {local_A}")
    local_B = [rects_sectors[r] for r in range(0, len(rects_sectors)) if r % 2 == 1][rank::size]
    print(f"[Rank {rank}] Размер локальной подгруппы B: {len(local_B)}, состав: {local_B}")
    func = boundary
    for iter_num in range(max_iter_schwarz):
        print(f"[Rank {rank}] Итерация {iter_num + 1}", flush=True)

        # ШАГ 1: SSOR на локальной подгруппе B
        for obj in local_B:
            obj.ssor(func, max_iter_ssor)

        # ШАГ 2: Отправка границ из B → A (всем процессам)
        data_B = serialize_objects(local_B)
        all_data_B = comm.allgather(data_B)

        # ШАГ 3: merge B → A
        for remote_B_data in all_data_B:
            remote_B_objects = deserialize_objects(remote_B_data)
            print(f"[Rank {rank}] Размер получаемой подгруппы B: {len(remote_B_objects)}, состав: {remote_B_objects}")
            for obj_a in local_A:
                for obj_b in remote_B_objects:
                    if obj_a.has_intersection(obj_b):
                        obj_a.merge_from_intersection_perimeter(obj_b)

        # ШАГ 4: SSOR на локальной подгруппе A
        for obj in local_A:
            obj.ssor(func, max_iter_ssor)

        # ШАГ 5: Отправка границ из A → B
        data_A = serialize_objects(local_A)
        all_data_A = comm.allgather(data_A)

        # ШАГ 6: merge A → B
        for remote_A_data in all_data_A:
            remote_A_objects = deserialize_objects(remote_A_data)
            for obj_b in local_B:
                for obj_a in remote_A_objects:
                    if obj_b.has_intersection(obj_a):
                        obj_b.merge_from_intersection_perimeter(obj_a)

        # ШАГ 7: Вычисление отклонения
        local_diffs = [obj_a.compute_norm_in_intersection(obj_b)
                       for obj_a in local_A
                       for obj_b in remote_B_objects
                       if obj_a.has_intersection(obj_b)]

        local_max_diff = max((d[0] for d in local_diffs), default=0)
        global_max_diff = comm.allreduce(local_max_diff, op=MPI.MAX)

        if rank == 0:
            print(f"[Global] Max diff: {global_max_diff:.2e}", flush=True)

        if rank == 0 and (iter_num < 5 or (iter_num + 1) % 10 == 0 or global_max_diff < EPS):
            all_objects = local_A + local_B
        else:
            all_objects = None

        # Собираем у всех
        serialized_local = pickle.dumps(local_A + local_B)
        gathered = comm.gather(serialized_local, root=0)

        if rank == 0:
            deserialized_all = []
            for ser in gathered:
                deserialized_all.extend(pickle.loads(ser))

            draw_field(deserialized_all, name=f"Итерация {iter_num + 1}, max_diff: {global_max_diff:.2e}")

        if global_max_diff < EPS:
            if rank == 0:
                print("Сходимость достигнута", flush=True)
            break

    plt.show(block=True)
