


import scipy.stats as stats


def studentt(
        df,
        confidence = 90,
        twosided=True
):
    """
    Student t distribution t[p; n]
    where p = 0.95 (90% two-sided)
    and n is degrees of freedom

    Arguments:

        df (integer):
            degrees of freedom
        confidence (integer):
            Desired confidence,
            a percent quoted as an integer.
            (Default: ``90``)
        twosided (boolean)
            whether to evaluate the
            t-distribution for two-sided
            confidence.
            (Default: ``True``)
    """
    if not 50 <= confidence <= 100:
        raise ValueError(f"confidence {confidence} is out of bounds (typical values: 80, 90, 95, etc.)")
    alpha = (100.0 - confidence)/100.0
    if twosided:
        alpha = alpha/2.0
    p = stats.t.ppf(
        q=1-alpha,
        df=df
    )
    return p






def Fnm(
        df1,
        df2,
        confidence = 90,
        twosided=False,
):
    """
    F(n,m) distribution
    commonly denoted F[p; n, m],
    for two degrees of freedom n, m.

    Arguments:

        df1 (integer):
            degrees of freedom, numerator
        df2 (integer):
            degrees of freedom, denominator
        confidence (integer):
            Desired confidence,
            a percent quoted as an integer.
            (Default: ``90``)
        twosided (boolean)
            whether to evaluate the
            distribution for two-sided
            confidence.
            (Default: ``False``)
    """
    if not 50 <= confidence <= 100:
        raise ValueError(f"confidence {confidence} is out of bounds (typical values: 80, 90, 95, etc.)")
    alpha = (100.0 - confidence)/100.0
    if twosided:
        alpha = alpha/2.0
    p = stats.f.ppf(
        q=1-alpha,
        dfn=df1,
        dfd=df2,
        # using defaults:
        loc=0.0,
        scale=1.0,
    )
    return p






