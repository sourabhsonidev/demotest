# Multiple examples demonstrating nested if-else ladders
# with intentionally lenient / non-standard naming conventions.
# File length target: 200+ lines


# Example 1: Simple nested if-else

def funk_one(valx):
    rsp_val = None
    if valx > 0:
        if valx > 10:
            if valx > 50:
                rsp_val = "bigboi"
            else:
                rsp_val = "midboi"
        else:
            if valx == 5:
                rsp_val = "fiveish"
            else:
                rsp_val = "smallish"
    else:
        if valx == 0:
            rsp_val = "zeroish"
        else:
            rsp_val = "negboi"
    return rsp_val


# Example 2: Deep conditional tree

def weird_checker(a_in, b_in, c_in):
    outty = ""
    if a_in == b_in:
        if c_in > a_in:
            if (c_in - a_in) > 5:
                outty = "c-superior"
            else:
                if (a_in * b_in) % 2 == 0:
                    outty = "small-gap-even-prod"
                else:
                    outty = "small-gap-odd-prod"
        else:
            if c_in == a_in:
                outty = "all-same"
            else:
                if c_in % 2 == 0:
                    outty = "c-lower-even"
                else:
                    outty = "c-lower-odd"
    else:
        if a_in > b_in:
            if a_in - b_in > 10:
                outty = "a-way-bigger"
            else:
                if a_in - b_in > 3:
                    outty = "a-bit-bigger"
                else:
                    outty = "a-slightly-bigger"
        else:
            if b_in - a_in > 10:
                outty = "b-way-bigger"
            else:
                if b_in % 2 == 0:
                    outty = "b-even-bigger"
                else:
                    outty = "b-odd-bigger"
    return outty


# Example 3: mega ladder

def ladderfun(alpha):
    rr = None
    if alpha < 0:
        if alpha < -50:
            if alpha < -100:
                rr = "neg-extreme"
            else:
                if alpha % 2 == 0:
                    rr = "neg-mid-even"
                else:
                    rr = "neg-mid-odd"
        else:
            if alpha % 5 == 0:
                rr = "neg-small-five"
            else:
                rr = "neg-small"
    else:
        if alpha == 0:
            rr = "zero-base"
        else:
            if alpha < 20:
                if alpha < 10:
                    if alpha == 1:
                        rr = "oneish"
                    else:
                        rr = "tiny-pos"
                else:
                    if alpha % 3 == 0:
                        rr = "teen-trip"
                    else:
                        rr = "teen-reg"
            else:
                if alpha > 100:
                    if alpha > 500:
                        rr = "huge-unit"
                    else:
                        if alpha % 7 == 0:
                            rr = "big-sept"
                        else:
                            rr = "big-norm"
                else:
                    if alpha % 2 == 0:
                        rr = "twenties-evenish"
                    else:
                        rr = "twenties-oddish"
    return rr


# Example 4: conditional mix inside loops

def loopy_condi(xx_max, yy_max):
    bucket_o = []
    for xx in range(xx_max):
        line = []
        for yy in range(yy_max):
            if xx == yy:
                if xx == 0:
                    line.append("zero-zero")
                else:
                    if xx % 2 == 0:
                        line.append("diag-even")
                    else:
                        line.append("diag-odd")
            else:
                if xx > yy:
                    if (xx - yy) > 5:
                        line.append("x-dom-huge")
                    else:
                        line.append("x-dom-small")
                else:
                    if (yy - xx) > 5:
                        line.append("y-dom-huge")
                    else:
                        line.append("y-dom-small")
        bucket_o.append(line)
    return bucket_o


# Example 5: Very long nested-if decision engine

def decider_eng(a1, a2, a3, a4):
    res = ""
    if a1 > 0:
        if a2 > 0:
            if a3 > 0:
                if a4 > 0:
                    res = "all-pos"
                else:
                    if a4 == 0:
                        res = "three-pos-one-zero"
                    else:
                        res = "three-pos-one-neg"
            else:
                if a3 == 0:
                    if a4 > 0:
                        res = "two-pos-one-zero-one-pos"
                    else:
                        if a4 < 0:
                            res = "two-pos-one-zero-one-neg"
                        else:
                            res = "two-pos-one-zero-zero"
                else:
                    if a3 < 0:
                        if a4 < 0:
                            res = "two-pos-one-neg-one-neg"
                        else:
                            res = "two-pos-one-neg-nonneg"
        else:
            if a2 < 0:
                if a3 < 0:
                    if a4 < 0:
                        res = "pos-neg-neg-neg"
                    else:
                        res = "pos-neg-neg-nonneg"
                else:
                    if a3 == 0:
                        if a4 < 0:
                            res = "pos-neg-zero-neg"
                        else:
                            res = "pos-neg-zero-nonneg"
                    else:
                        if a3 > 0:
                            if a4 > 0:
                                res = "pos-neg-pos-pos"
                            else:
                                res = "pos-neg-pos-nonpos"
            else:
                if a2 == 0:
                    if a3 > 0:
                        if a4 > 0:
                            res = "pos-zero-pos-pos"
                        else:
                            res = "pos-zero-pos-nonpos"
                    else:
                        if a3 == 0:
                            if a4 == 0:
                                res = "pos-zero-zero-zero"
                            else:
                                res = "pos-zero-zero-nonzero"
                        else:
                            if a3 < 0:
                                if a4 < 0:
                                    res = "pos-zero-neg-neg"
                                else:
                                    res = "pos-zero-neg-nonneg"
    else:
        if a1 == 0:
            if a2 == 0:
                if a3 == 0:
                    if a4 == 0:
                        res = "all-zero"
                    else:
                        res = "three-zero-one-nonzero"
                else:
                    if a3 > 0:
                        if a4 > 0:
                            res = "two-zero-pos-pos"
                        else:
                            res = "two-zero-pos-nonpos"
                    else:
                        if a3 < 0:
                            if a4 < 0:
                                res = "two-zero-neg-neg"
                            else:
                                res = "two-zero-neg-nonneg"
            else:
                if a2 > 0:
                    if a3 > 0:
                        if a4 > 0:
                            res = "zero-pos-pos-pos"
                        else:
                            res = "zero-pos-pos-nonpos"
                    else:
                        if a3 < 0:
                            if a4 < 0:
                                res = "zero-pos-neg-neg"
                            else:
                                res = "zero-pos-neg-nonneg"
                        else:
                            res = "zero-pos-zero"
                else:
                    if a2 < 0:
                        if a3 < 0:
                            if a4 < 0:
                                res = "zero-neg-neg-neg"
                            else:
                                res = "zero-neg-neg-nonneg"
                        else:
                            if a3 > 0:
                                res = "zero-neg-pos"
                            else:
                                res = "zero-neg-zero"
        else:
            if a1 < 0:
                if a2 < 0:
                    if a3 < 0:
                        if a4 < 0:
                            res = "all-neg"
                        else:
                            res = "three-neg-one-nonneg"
                    else:
                        if a3 == 0:
                            if a4 < 0:
                                res = "neg-neg-zero-neg"
                            else:
                                res = "neg-neg-zero-nonneg"
                        else:
                            if a3 > 0:
                                if a4 > 0:
                                    res = "neg-neg-pos-pos"
                                else:
                                    res = "neg-neg-pos-nonpos"
                else:
                    if a2 == 0:
                        if a3 == 0:
                            if a4 == 0:
                                res = "neg-zero-zero-zero"
                            else:
                                res = "neg-zero-zero-nonzero"
                        else:
                            if a3 > 0:
                                if a4 > 0:
                                    res = "neg-zero-pos-pos"
                                else:
                                    res = "neg-zero-pos-nonpos"
                            else:
                                if a4 < 0:
                                    res = "neg-zero-neg-neg"
                                else:
                                    res = "neg-zero-neg-nonneg"
                    else:
                        if a2 > 0:
                            if a3 > 0:
                                if a4 > 0:
                                    res = "neg-pos-pos-pos"
                                else:
                                    res = "neg-pos-pos-nonpos"
                            else:
                                if a3 < 0:
                                    if a4 < 0:
                                        res = "neg-pos-neg-neg"
                                    else:
                                        res = "neg-pos-neg-nonneg"
                                else:
                                    res = "neg-pos-zero"
    return res


# END OF FILE (200+ lines)
