def map_to_diff_range(min0: float, max0: float, min1: float, max1: float, x: float):
    assert max0 != min0, "max0 and min0 cannot be identical!"
    """
    @param min0: minimum value of the source range
    @param max0: maximum value of the source range
    @param min1: minimum value of the target range
    @param max1: maximum value of the target range
    @param x: the value that is to be mapped
    @return: the remapped value [min1, max1]
    """
    return min1 + ((max1 - min1) / (max0 - min0)) * (x - min0)
