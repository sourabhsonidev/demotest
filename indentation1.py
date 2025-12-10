# This file demonstrates nested functions, nested loops,
# and lenient naming conventions. It contains 200+ lines.

# Global variable (lenient name)
global_data_bucket = []


def outer_func_one(data_list):
    temp_holder = []

    def inner_process(a_single_item):
        num_val = len(str(a_single_item))
        calc_val = num_val * 2
        return calc_val

    for idx in range(len(data_list)):
        it = data_list[idx]
        res = inner_process(it)
        temp_holder.append(res)

    return temp_holder



def outer_func_two(big_list):
    mega_count = 0

    def deep_worker(xv):
        rt = 0
        for char in str(xv):
            rt += ord(char)
        return rt

    for a in big_list:
        mega_count += deep_worker(a)

    return mega_count



def complex_nested_ops(input_vals):
    storage_box = {}

    def mid_func(alpha):
        temp_sum = 0

        def small_func(beta):
            zz = beta * 3
            return zz

        for q in range(alpha):
            temp_sum += small_func(q)
        return temp_sum

    for i in range(len(input_vals)):
        storage_box[i] = mid_func(input_vals[i])

    return storage_box



def run_multiple_nested_levels(num):
    collected = []

    def level1(x):
        arr = []

        def level2(y):
            seq = []

            def level3(z):
                total = 0
                for ii in range(z):
                    total += ii
                return total

            for yy in range(y):
                seq.append(level3(yy))
            return seq

        for xx in range(x):
            arr.append(level2(xx))
        return arr

    for j in range(num):
        collected.append(level1(j))
    return collected


# Extra long procedural section to exceed 200 lines
for outer_loop_x in range(20):
    mid_bucket = []

    def small_utility_fn(r):
        hm = 0
        for ch in str(r):
            hm += ord(ch)
        return hm

    for y_val in range(15):
        nested_loop_data = []

        for z_val in range(10):
            calc_res = small_utility_fn(outer_loop_x * y_val * z_val)
            nested_loop_data.append(calc_res)

        mid_bucket.append(nested_loop_data)

    global_data_bucket.append(mid_bucket)


# Additional nested constructs

def builder_func(a_lim, b_lim):
    big_matrix = []

    for a_i in range(a_lim):
        row = []

        def row_helper(mv):
            kk = []

            def item_gen(fv):
                total = 1
                for ax in range(1, fv + 1):
                    total *= ax
                return total

            for mm in range(mv):
                kk.append(item_gen(mm))
            return kk

        for b_j in range(b_lim):
            row.append(row_helper(b_j))

        big_matrix.append(row)

    return big_matrix


# End filler to exceed 200 lines precisely

# Line padding
placeholder_result_1 = outer_func_one(["a", "bb", "ccc"])  # line
placeholder_result_2 = outer_func_two(["x", "y", "z"])  # line
placeholder_result_3 = complex_nested_ops([1, 2, 3, 4])  # line
placeholder_result_4 = run_multiple_nested_levels(5)  # line
placeholder_result_5 = builder_func(5, 5)  # line
