




from math import sqrt
from os.path import \
    exists as os_path_exists
from numpy import \
    zeros as numpy_zeros, \
    savetxt as numpy_savetxt

from .experiment_impl.experiment import \
    Experiment
from .experiment_impl.stats_tables import \
    studentt, Fnm




class OneWay(Experiment):
    """
    Output of a 1-parameter experiment with a single response,
    having a range of size N values,
    over r repetitions, evaluating confidence.

    The inputs are two files, whose names are in the default case,
    ``response.csv`` and ``specification.csv``.

    There is 1 line in the parameter specification file.
    Values 1 through N, called levels
    (other names may be more appropriate, it depends on the data.)
    The format of the line is::

        <name>, <level1>, <level2>, <level3>, ..., <levelN>

    Levels do not need to be numeric, but if it is numeric,
    it is probably wise to place levels in ascending order,
    as this order will be reflected in the output.
    The name assigns a label to the parameter.

    The format for the response file is a header line,
    followed by comma separated values.
    There are N nonempty lines, one for each level.
    Each line is::

        <value>,<value>,<value>

    with the ith line having r columns, where r is the
    number of repetitions. The number of repetitions
    can vary with the level. (This is useful if one or more
    experiments needs to be thrown out.)

    .. note::

        It is assumed there is *always* a header line
        (containing the response name) in the response file.

    Parameters:

        specification (optional string):
            The parameter specification
            An absolute or relative path may be passed
            in place of a string, however, this path's filename
            must have a .csv ending. Otherwise,
            the processing proceeds as though it is a raw string.
        verbose (boolean):
            Whether to run with verbose output. Default: False

    """

    # Ideas:
    # - Twokr with r allowed to vary between configurations
    # - Ntwor, two factor full factorial experiment
    # - Nkr, full general factorial experiment
    # - Fractional Twokr experiment

    # todo quiet/automated delivery of model output

    def __init__(
            self,
            specification = None,
            verbose = False,
    ):
        super().__init__(
            specification=specification,
            verbose=verbose,
        )
        self.values = self.Values()
        # style parameters
        # ensure walk_full % delta == 0
        self.full_walk = 60
        self.delta_walk = 2
        # drawing tmps - this is not convenient in Python
        self.draw_variatn_noise = 0.0
        self.draw_variatn_current = 0.0



    def start(
            self,
            location,
            response = None,
            confidences = None,
            show_center = None,
            scalar_levels = False,
    ):
        """
        Start the analysis with the provided response.
        This method can be called any number of times
        provided the parameter specification is unchanged.

        .. note::

            The location selected should be an empty directory,
            or a directory that is designated for output.
            It can be cleaned in order to make room for
            new output, as this is convenient when re-running
            experiments. TL;DR: do not select a location
            where you have existing files.

        Arguments:

            response (optional string):
                The response, or experimental measurements.
                An absolute or relative path may be passed
                in place of a string, however, this path's filename
                must have a .csv ending. Otherwise,
                the processing proceeds as though it is a
                raw response string.
            location (:any:`Location`):
                A location for the experiment.
                If the response is not provided,
                it will be searched for in this location
                as "response.csv"
            confidences (list of integer):
                todo
            show_center (boolean):
                todo
            scalar_levels (boolean):
                todo

        """
        self.location = location
        self.location.create_path()
        response_ = self.location.filename("response.csv")
        if response is not None:
            if response[-4:] != '.csv':
                self.data = response.strip()
            else:
                with open(response, 'r') as f:
                    self.data = f.read().strip()
            write_response = True
        else:
            # > look for response (repeat run)
            if os_path_exists(response_):
                # > proceed with response
                with open(response_, 'r') as f:
                    self.data = f.read().strip()
            else:
                raise ValueError(f"Response not found.")
            write_response = False
        # > clean path directory
        # todo skipping for now
        self.location.clean(
            skip=["response.csv", "specification.csv"],
        )
        self.log(f"k {self.k} r {self.r} N {self.N}")
        self.log(f'~(N^1)r experiment~\nN = {self.N}, r = {self.r}')
        self._init()
        self._read_data()
        self._compute_values()
        self._make_artifacts(
            response_path=response_ if write_response else None,
            confidences = confidences,
            show_center = show_center,
            scalar_levels = scalar_levels,
        )



    def _make_artifacts(
            self,
            response_path,
            confidences,
            show_center,
            # todo utilize 'scalar_levels' or deprecate
            scalar_levels,
    ):
        # > write artifacts to allow re-runs
        if response_path is not None:
            with open(response_path, 'w') as f:
                f.write(self.data)
            specification_path = self.location.filename("specification.csv")
            with open(specification_path, 'w') as f:
                f.write(self.specification)
        handle = f"{self.pnames[0]}--{self.response_name}"
        filename = self.location.filename(
            f"{handle}.summary.txt"
        )
        summary = self.summary()
        with open(filename, 'w') as f:
            f.write(summary)
        if self.r > 1:
            # > write 90-95-99 artifacts
            filename = self.location.filename(
                f"{handle}.90-95-99.dat",
            )
            numpy_savetxt(fname=filename, X=self.X)
            filename = self.location.filename(
                f"{handle}.90-95-99.png",
            )
            self.fig.errorbarseries(
                filename=filename,
                X=self.X,
                title=None,
                inlabel=self.pnames[0],
                inticklabels=self.plvl[0],
                outlabel=self.response_name,
                center=self.values.mu,
                fmt_999=True,
            )
            # Python bobble
            confidences_ = confidences if confidences is not None else []
            for C in confidences_:
                if C in [90,95,99]:
                    filename = self.location.filename(
                        f"{handle}.{C}.png",
                    )
                    self.fig.errorbarseries(
                        filename=filename,
                        X=self.X,
                        title=None,
                        inlabel=self.pnames[0],
                        inticklabels=self.plvl[0],
                        outlabel=self.response_name,
                        center=self.values.mu if show_center or show_center is None else None,
                        error_idx=2 if C == 90 else 4 if C == 95 else 6,
                    )
                else:
                    # > another confidence request - rare case (???)
                    raise NotImplementedError
        else:
            # > no repetitions - a slightly different artifact procedure
            filename = self.location.filename(
                f"{handle}.dat",
            )
            numpy_savetxt(fname=filename, X=self.X)
            filename = self.location.filename(
                f"{handle}.png",
            )
            self.fig.errorbarseries(
                filename=filename,
                X=self.X,
                title=None,
                inlabel=self.pnames[0],
                inticklabels=self.plvl[0],
                outlabel=self.response_name,
                # only show center if explicitly requested
                center=self.values.mu if show_center else None,
            )





    class Values:

        def __init__(self):
            # the grand mean
            self.mu = 0.0
            self.SS0 = 0.0
            self.SSA = 0.0
            self.SSY = 0.0
            self.SSE = 0.0
            # standard deviation of errors
            self.se = 0.0

        def Avarn(self):
            if self.SSY == 0.0 and self.SS0 == 0.0:
                print(f"[OneWay::Values::Avarn] [Warning] SSY, SS0 are both zero.")
                return 1000.0
            return self.SSA/(self.SSY-self.SS0)*100



    def _compute_values(self):
        """
        compute SS, SSY, SST, SSE, se.

        SS0 is the total degrees of freedom times the grand mean squared.

        SSA is the sum from 1 to N of the level-specific mean
        times the number of repetitions available in that level, r_i.

        SSY is the sum of squares of all raw measurements.

        SSE is the remainder when SSA + SS0 is differenced from SSY.

        se is the standard deviation of errors, sqrt(SSE/(DF-N))
        where DF is the total degrees of freedom (number of summed terms
        in SSY)

        """
        # > compute grand mean
        self.values.mu = 0.0
        for i in range(self.N):
            self.values.mu += self.esmpls[i].r()*self.esmpls[i].response
        self.values.mu /= self.DF
        # SS0
        mu0 = self.values.mu
        DF = self.DF
        self.values.SS0 = DF*mu0*mu0
        # SSY and SSA
        self.values.SSY = 0.0
        self.values.SSA = 0.0
        for i in range(self.N):
            alpha = self.esmpls[i].response - mu0
            r = self.esmpls[i].r()
            for rep in range(r):
                y = self.esmpls[i].raw_data[rep]
                self.values.SSY += y*y
            self.values.SSA += r*alpha*alpha
        # SSE, se, s
        if self.r > 1:
            self.values.SSE = self.values.SSY - self.values.SS0 - self.values.SSA
            if self.values.SSE < 0:
                if self.values.SSE < -1e-8:
                    raise ValueError
                self.values.SSE = 0.0
            self.values.se = sqrt(self.values.SSE/(self.DF - self.N))
        else:
            self.values.SSE = 0.0
            self.values.se = 0.0
        if self.r > 1:
            self.X = numpy_zeros((self.N, 1+1+2*3))
            # > populate
            for i in range(self.N):
                row = numpy_zeros(1+1+2*3)
                # todo optionally set the 'x' value here, though it is not computationally significant
                row[0] = float(i)
                row[1] = self.esmpls[i].response
                for j, C in enumerate([90, 95, 99]):
                    studentC = studentt(
                        df=self.DF - self.N,
                        confidence=C,
                        # todo review
                        twosided=True,
                    )
                    sdev = self.values.se*sqrt((self.N-1)/self.DF)
                    delta = sdev*studentC
                    # it is the same on both sides:
                    minus = delta
                    plus = delta
                    row[2+2*j] = minus
                    row[2+2*j+1] = plus
                self.X[i,:] = row
        else:
            # > no repetitions
            self.X = numpy_zeros((self.N, 1+1))
            for i in range(self.N):
                row = numpy_zeros(1+1)
                # todo optionally set the 'x' value here, though it is not computationally significant
                row[0] = float(i)
                row[1] = self.esmpls[i].response
                self.X[i,:] = row




    def F_test(self, confidence):
        out = ""
        F = Fnm(
            df1=self.N-1,
            df2=self.DF - self.N,
            confidence=confidence,
            twosided=False,
        )
        MSA = self.values.SSA/(self.N-1)
        MSE = self.values.SSE/(self.DF-self.N)
        Fvariable = MSA/MSE
        out += f"[ANOVA::F-test] The null hypothesis states that samples in all levels are sampled \nfrom a population with the same population-wide mean response.\n"
        if Fvariable > F:
            out += f"[ANOVA::F-test] {Fvariable} > {F}\n"
            out += f"[ANOVA::F-test] computed > table\n"
            out += f"[ANOVA::F-test] variance in level-wise means of {self.pnames[0]} is judged significant.\n"
            out += f"[ANOVA::F-test] The null hypothesis is rejected."
        else:
            out += f"[ANOVA::F-test] {Fvariable} ≤ {F}\n"
            out += f"[ANOVA::F-test] computed ≤ table\n"
            out += f"[ANOVA::F-test] variance in level-wise means of {self.pnames[0]} is judged not significant.\n"
            out += f"[ANOVA::F-test] The null hypothesis is not rejected.\n"
        return out



    def Kruskal_Wallis_test(self, confidence):
        out = ""
        # out += f"[Kruskal-Wallis-test] todo\n"
        return out



    def summary(self):
        out = ""
        confidences = [90]
        for C in confidences:
            if self.r > 1:
                studentC = studentt(
                    df=self.DF - self.N,
                    confidence=C,
                    twosided=True,
                )
                out += f"Note: 90% confidence is used. If any confidence interval contains 0.0, \nthe result is insignificant with {C}% certainty. \n\n"
            else:
                studentC = None
                out += "Zero repetitions of the experiment. \n\n"
            mu = self.values.mu
            out += f"The total average is {mu}\n"
            delta = 0.0
            if studentC:
                sdev = self.values.se/sqrt(self.DF)
                delta = sdev*studentC
                out += f"...confidence interval {mu-delta}, {mu+delta}\n"
                # > reset delta for effects
                sdev = self.values.se*sqrt((self.N-1)/self.DF)
                delta = sdev*studentC
            for i in range(self.N):
                lname = self.plvl[0][i]
                alpha = self.esmpls[i].response-mu
                out += f"The effect of {self.pnames[0]} level {lname} is {alpha}\n"
                if studentC:
                    out += f"...confidence interval {alpha-delta}, {alpha+delta}\n"
            if self.r > 1:
                out += f'---\n'
                Avarn = self.values.Avarn()
                out += f"[ANOVA] The effect of {self.pnames[0]} accounts for {Avarn}% of variation.\n"
                out += f"[ANOVA] The rest of the variation is noise: {100-Avarn}% of variation.\n"
                # todo contrasts: see e.g. Jain pp. 336-337
                # out += f"[ANOVA] Contrasts:\n"
                # todo tests: for now just throw the kitchen sink at it
                out += self.F_test(confidence=C)
                out += self.Kruskal_Wallis_test(confidence=C)
            out += "\n"
        out += "......................Input Data (received).................\n"
        out += "idx, rep, response, raw_data[rep]\n"
        for i in range(self.N):
            expt = self.esmpls[i]
            for rep in range(self.r):
                if expt.raw_data[rep] is not None:
                    out += f"{i}, {rep}, {expt.response}, {expt.raw_data[rep]}\n"
        out += ".............................Response Data.............................\n"
        out += f"level, response\n"
        for i in range(self.N):
            out += f"{self.pnames[0]}_{self.plvl[0][i]}, {self.esmpls[i].response}\n"
        return out






