


import numpy as np


def _get_assist_points(dimension, Nassist):
    if Nassist >= 64:
        # if dimension is large enough, we must generate randomly
        generate_random = True
    else:
        if Nassist > 2**(dimension-1):
            # we wish to add more than 50% of the points,
            # or the request exceeds the possible number of points 2^dim.
            generate_random = False
        else:
            # we only wish to add < 50% of the points
            generate_random = True
    # lazy: for now, always use choose_random procedure.
    generate_random = True
    if generate_random:
        # The intended use case assumes that assist points
        # populate a small fraction of the possible space.
        # if this assumption is invalidated, the procedure is
        # inefficient and subject to failure if dimension is large enough.
        points = np.random.uniform(size=(Nassist, dimension))*2 - 1
        points = np.where(points >= 0, 1, -1)
        # the penalty for this inexpensive procedure is we have to remove dups
        bad_points = []
        safety_limit = 40
        safety = 0
        while True:
            # redo bad points
            for i in bad_points:
                point = np.random.uniform(size=(1,dimension))*2-1
                point = np.where(point >= 0, 1, -1)
                points[i,:] = point
            bad_points = []
            # repopulate bad points
            for i in range(points.shape[0]):
                for j in range(i+1, points.shape[0]):
                    if np.allclose(points[i,:], points[j,:]):
                        # i is bad
                        bad_points.append(i)
                    else:
                        # i is not bad
                        pass
            if len(bad_points) == 0:
                break
            safety += 1
            if safety == safety_limit:
                # This is a fixable implementation issue.
                raise ValueError("[get_assisted_axes] unique assist points could not be generated!")
    else:
        # generate the full set deterministically
        # and randomly remove points, if necessary,
        # rather than generating points randomly.
        raise NotImplementedError
    return points, safety


def get_assisted_axes(dimension, Nh, verbose=False):
    """
    The sample set points have uniform radius.
    Let $d$ be the dimension, $N_h$ be the `Nh` argument.
    This radius is $\sqrt{d}$ to prevent underflow
    when dimension is large. The caller should scale
    as desired based on this convention.

    The sample set size may be truncated in cases where
    dimension is small, but in the nominal situation
    the sample set size is $(2+N_h)*d$.

    :param dimension:
    :param Nh: Assist points factor.
    This factor multiplicatively scales with dimension.
    In the ordinary case (when there are sufficient points available)
    the number of assist points is dimension*Nh.
    When there are insufficient points available this value is truncated.
    :param verbose: whether to print a short diagnostic string
    :return: assisted axes sample set
    """
    if Nh*dimension >= 64:
        Nassist = Nh*dimension
    else:
        # dimension is small enough to take powers of two
        Nassist = min(Nh*dimension, 2**dimension)
    np.random.seed(300)
    out = np.zeros((2*dimension+Nassist,dimension))
    for i in range(dimension):
        out[2*i,i] = np.sqrt(dimension)
        out[2*i+1,i] = -np.sqrt(dimension)
    assist_points, safety = _get_assist_points(dimension=dimension, Nassist=Nassist)
    if verbose:
        print(f"dim {dimension} Nh {Nh} Nassist {Nassist} points {out.shape} assist_points {assist_points.shape} safety {safety}")
    out[2*dimension:,:] = assist_points
    return out

