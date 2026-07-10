

from os.path import (
    join as os_path_join,
)
from .experiment_impl.experiment import (
    Experiment,
)
from .experiment_impl.stats_tables import (
    studentt,
)
from math import sqrt, fabs

from physixlib import (
    popcount,
    get_bit,
)



class Twokr(Experiment):
    """
    Output of a k-parameter experiment with a single response,
    and uniformly binary input ranges
    over r repetitions, evaluating confidence
    and computing ANOVA statistics.
    An experiment with multiple responses can be
    handled with multiple instantiations of the procedure.

    There are k lines in the parameter specification file.
    The format for each line is::

        <name>, <level>, <level>

    where name is a string (a name of the parameter,
    also called a factor), and levels are scalar values.
    The first level must always be the lower (numerically)
    of the two.

    .. note::

        The parameter specification file
        determines the order of the parameters,
        and the same order must be observed in the
        data file. This will now be discussed further.

    The response file stores measurements
    in increasing order based on the binary-number
    stamp defining the parameter setting.
    Purely by convention, the order of the variables
    is mapped to a **most-to-least**-significant ordering of an
    increasing binary integer sequence.
    Example:
    The parameters are A, B, C.
    This is their order, given in the parameter specification file.
    This assigns values 0, 1, 2 to these respectively, and these
    are viewed as place values in a binary number.
    The order of the settings in the parameter specification file
    determines Low, High settings. (This is the order precisely, Low, High.)
    The data for the experiment is stored as::

        1.23 # is the response when A low, B low, C low
        2.34 # is the response when A low, B low, C high
        3.45 # is the response when A low, B high, C low
        4.56 # is the response when A low, B high, C high
        5.67 # is the response when A high, B low, C low
        # etc.

    There is a total of eight rows in this `k=3` example.
    In all there are are a total of `2**k` lines, plus a header line.
    If low/high is mapped to binary digits, then
    the rows form an ordered list 000, 001, 010, 011, 100, 101, 110, 111.
    When written this way, in base two, the places of digits correspond
    (in this example) to {A, B, C}.
    None of this depends in any way on the endianness of
    your system's computer architecture!

    When multiple repetitions `r > 1` are performed, these form columns.
    Each line then becomes::

        <value>,<value>,<value>

    with every line having `r` columns. A varied number of repetitions
    is not supported.

    .. note::

        It is assumed there is *always* a header line
        with the response name in the response file.

    Parameters:

        specification (optional string):
            The parameter specification
            given directly as a string.
        verbose (boolean):
            Whether to run with verbose output. Default: False

    """

    def __init__(
            self,
            specification = None,
            verbose = False,
    ):
        super().__init__(
            specification=specification,
            verbose=verbose,
        )
        # todo there's a bug/mistake in the impl where
        #  the specification is unpacked twice, once by
        #  Experiment and once here. Fix when there is time.
        #  Experiment should (?) do the same work for both Twokr and Noner.
        lines = self.specification.split('\n')
        self.k = len(lines)
        # if r < 2:
        #     raise NotImplementedError(f"Number of repetitions {r} is too small, it is not supported.")
        if self.k >= 32:
            # just a sanity check
            raise ValueError("k is too large.")
        self.plvl = []
        self.pnames = []
        for _ in range(self.k):
            self.plvl.append([0.0, 0.0])
            self.pnames.append('')
        self.data = None
        self.student90 = None
        self.model = self.Model(N_to_the_k=self.N_to_the_k)
        self.values = self.Values(N_to_the_k=self.N_to_the_k)
        # style parameters
        # ensure walk_full % delta == 0
        self.full_walk = 60
        self.delta_walk = 2
        # drawing tmps - this is not convenient in Python
        self.draw_variatn_noise = 0.0
        self.draw_variatn_current = 0.0
        self.r = None
        self.out = None
        # > read pspec
        lines = self.specification.split('\n')
        for pidx, line in enumerate(lines):
            items = line.split(',')
            self.pnames[pidx] = items[0].strip()
            level0, level1 = items[1], items[2]
            self.plvl[pidx] = [float(level0), float(level1)]
        # self.log(f'{self.plvl}')
        # self.log(f'{self.pnames}')



    def start(
            self,
            response,
    ):
        """
        Start the analysis with the provided response.
        This method can be called any number of times
        provided the parameter specification is unchanged.

        Arguments:

            response (string):
                The response, or experimental measurements.
                An absolute or relative path may be passed
                in place of a string, however, this path's filename
                must have a .csv ending. Otherwise,
                the processing proceeds as though it is a
                raw response string.

        """
        # todo variable confidence (now fixed at 90)

        # todo store response.{response_name}.csv

        self.out = ""
        if response[-4:] != '.csv':
            self.data = response.strip()
        else:
            with open(response, 'r') as f:
                self.data = f.read().strip()
        line = self.data.split('\n')[1]
        r = len(line.split(','))
        self.r = r
        self.student90 = studentt(
            df=self.N_to_the_k*(r-1),
            confidence=90,
            twosided=True,
        )
        self.esmpls = []
        for _ in range(self.N_to_the_k):
            self.esmpls.append(self.Experiment(r=r))
        if self.v:
            self.log(f"k {self.k} r {self.r} N_to_the_k {self.N_to_the_k}")
        out = f'~(2^k)r experiment~\nk = {self.k}, r = {self.r}'
        self.log(out, always=True)
        self.out += out
        self.read_data()
        out = f'response: {self.response_name}'
        self.log(out, always=True)
        self.out += out
        self.compute_model()
        self.compute_values()
        out = self.make_report()
        self.log(out, always=True)
        self.out += out
        with open(os_path_join(self.path, 'log.txt'), 'w') as f:
            f.write(self._log)


    class Experiment:

        def __init__(
                self,
                r,
        ):
            self.response = 0.0
            self.raw_data = []
            for _ in range(r):
                self.raw_data.append(0.0)


    class Model:

        def __init__(
                self,
                N_to_the_k,
        ):
            self.q = []
            for _ in range(N_to_the_k):
                self.q.append(0.0)


    class Values:

        def __init__(
                self,
                N_to_the_k,
        ):
            # Note: SS0 is SS[0]
            self.SS = []
            for _ in range(N_to_the_k):
                self.SS.append(0.0)
            self.SSY = 0.0
            self.SST = 0.0
            self.SSE = 0.0
            # standard deviation of errors
            self.se = 0.0
            # standard deviation of effects
            self.s = 0.0



    def read_data(self):
        """
        Read the individual responses into experiments
        and populate names and levels of parameters.
        """
        lines = self.data.split('\n')
        # We assume there is *always* a header line containing the response name.
        self.response_name = lines[0].strip()
        self.log(f"response: {self.response_name}")
        for i, line in enumerate(lines[1:]):
            smpl = self.esmpls[i]
            responses = line.split()
            for response in responses:
                trials = response.split(',')
                for j, trial in enumerate(trials):
                    trl = float(trial)
                    smpl.raw_data[j] = trl
                    smpl.response += trl
                smpl.response /= self.r
            self.log(f"smpl {i}: {smpl.raw_data}")
            self.log(f"response {i}: {smpl.response}")


    def compute_model(self):
        """
        Find all q model parameters.

        """
        for j in range(self.N_to_the_k):
            self.model.q[j] = 0.0
            for i in range(self.N_to_the_k):
                sij = self.sign(i, j)
                self.model.q[j] += sij*self.esmpls[i].response
            self.model.q[j] /= self.N_to_the_k
            self.log(f"q_j = {self.model.q[j]}")



    def compute_values(self):
        """
        compute SS, SSY, SST, SSE, se, s.

        """
        # SS[j]
        for j in range(self.N_to_the_k):
            q = self.model.q[j]
            self.values.SS[j] = self.N_to_the_k*self.r*q*q
        # SSY
        self.values.SSY = 0.0
        for i in range(self.N_to_the_k):
            for rep in range(self.r):
                x = self.esmpls[i].raw_data[rep]
                self.values.SSY += x*x
        # SST
        self.values.SST = self.values.SSY - self.values.SS[0]
        # SSE, se, s
        if self.r > 1:
            self.values.SSE = self.values.SST - self.sum_SSnz()
            if self.values.SSE < 0:
                if self.values.SSE < -1e-8:
                    raise ValueError
                self.values.SSE = 0.0
            self.values.se = sqrt(self.values.SSE/(self.N_to_the_k*(self.r-1)))
            self.values.s = self.values.se/sqrt(self.N_to_the_k*self.r)
        else:
            self.values.SSE = 0.0
            self.values.se = 0.0
            self.values.s = 0.0



    def state_confidence(self, khot):
        if self.r == 1:
            return ""
        q = self.model.q[khot]
        s90 = self.student90
        s = self.values.s
        low = q - s90*s
        high = q - s90*s
        return f"....90% confidence interval ({low}, {high}).\n"


    def response(self, idx, rep=None):
        """
        Response at case index `idx` and optional repetition number `rep`.
        If no repetition number is given, the (average) response is given,
        otherwise the raw input response is given.
        :param idx: integer
        :param rep: integer (default: None)
        :return: scalar
        """
        smpl = self.esmpls[idx]
        if rep is not None:
            response = smpl.raw_data[rep]
        else:
            response = 0
            nrep = len(smpl.raw_data)
            for rep in range(nrep):
                response += smpl.raw_data[rep]
            response /= nrep
        return response


    def effect(self, idx):
        """
        Effect at case index `idx`
        :param idx:
        :return: scalar
        """
        return self.model.q[idx]


    def variation(self, idx):
        """
        Variation at case index `idx`
        :param idx:
        :return: scalar
        """
        return self.values.SS[idx]/self.values.SST*100.0


    def state_variation(self, khot):
        pct = self.variation(khot)
        return f"This accounts for {pct:2.3}% of all variation in {self.response_name}.\n"



    def draw_effect(self, khot):
        """
        The goal is to measure the effect as a percentage of the mean.
        In general, an effect could have a percent of the mean higher than 100%,
        e.g. if mean is 4 and effect is, say, +5.
        However, this would probably not occur often in practice,
        and if it did occur it might indicate that your model is wrong and you need to rethink.

        :param khot:
        :return:
        """
        delta = self.delta_walk
        full_walk = self.full_walk
        mean = self.model.q[0]
        diff = self.model.q[khot]
        pct = diff/mean*full_walk
        out = '\t\t'
        # ......GOAL: if pct == 0.0, never draw an arrow........
        if pct < -full_walk:
            out += '<!!'
            for i in range(full_walk//delta):
                out += '-'
        else:
            # draw the arrow where
            # i <= pct && i+D > pct
            # negate this:
            # i > pct || i+D <= pct
            out += '['
            i = -full_walk
            while i < 0:
                if i > pct:
                    out += '-'
                elif i <= pct - delta:
                    out += '_'
                else:
                    out += '<'
                i += delta
        out += '|'
        if pct > full_walk:
            out += '!!>'
        else:
            # draw arrow when
            # i >= pct && i-D < pct
            # as before.
            # negating gives
            # i < pct || i-D >= pct
            i = delta
            while i <= full_walk:
                if i < pct:
                    out += '-'
                elif i >= pct + delta:
                    out += '_'
                else:
                    out += '>'
                i += delta
            out += ']'
        out += f' {fabs(pct)}% of mean\n'
        return out

    def draw_variation_init(self):
        self.draw_variatn_noise = self.values.SSE/self.values.SST*2.0*self.full_walk
        self.draw_variatn_current = 0.0


    def draw_variation(self, khot):
        full_walk = self.full_walk
        delta = self.delta_walk
        variatn_noise = self.draw_variatn_noise
        variatn = (self.values.SS[khot]/self.values.SST)*2.0*full_walk
        variatn_old = self.draw_variatn_current
        self.draw_variatn_current += variatn
        variatn_current = self.draw_variatn_current
        out = '\t\t['
        # round up on the way up
        i = delta
        while i <= full_walk:
            if i < variatn_noise:
                out += 'x'
            elif variatn_old < i <= variatn_current + delta/2.0:
                out += '*'
            else:
                out += '_'
            i += delta
        # round down on the way down
        i = full_walk + delta
        while i <= 2.0*full_walk:
            if i < variatn_noise:
                out += 'x'
            elif variatn_old - delta/2.0 < i <= variatn_current:
                out += '*'
            else:
                out += '_'
            i += delta
        out += f"] {self.values.SS[khot]/self.values.SST*100.0}% of variation\n"
        return out




    def make_report(self):
        self.draw_variation_init()
        N_to_the_k = self.N_to_the_k
        s90 = self.student90
        values = self.values
        s = values.s
        out = ""
        out += f"response := average {self.response_name} over all repetitions\n"
        out += f"response[rep] := {self.response_name} at repetition rep (raw data input)\n"
        out += "......................Input Data (received).................\n"
        out += "khot_index, factor, response, rep, response[rep]\n"
        for idx in range(N_to_the_k):
            smpl = self.esmpls[idx]
            for rep in range(self.r):
                out += f"{idx}, {self.get_string(idx)}, {smpl.response}, {rep}, {smpl.raw_data[rep]}\n"
        if self.v:
            out += "...........................Debug Results...........................\n"
            out += "SSY, SST, SSE, se, s\n"
            out += f"{values.SSY}, {values.SST}, {values.SSE}, {values.se}, {s}\n"
            out += "khot_index, factor, effect, SS, %age of variation, lowerCI, upperCI\n"
            for khotidx in range(N_to_the_k):
                q = self.model.q[khotidx]
                out += f"{khotidx}, {self.get_string(khotidx)}, {values.SS[khotidx]}, {values.SS[khotidx]/values.SST*100.0}, {q - s90*s}, {q + s90*s}\n"
        out += "...........................Results...........................\n"
        out += f"factor, response, effect\n"
        for smpli in range(N_to_the_k):
            out += f"{self.pnames[0]}_{self.plvl[0][get_bit(smpli, 0)]}"
            for i in range(1, self.k):
                out += f"_{self.pnames[i]}_{self.plvl[i][get_bit(smpli, i)]}"
            avg = 0.0
            for rep in range(self.r):
                avg += self.esmpls[smpli].raw_data[rep]
            avg /= self.r
            out += f", {avg}, {self.model.q[smpli]}\n"
        out += ".............................Digest..................................\n"
        if self.r > 1:
            out += "\nNote: if any confidence interval contains 0.0, the result is insignificant. \n\n"
            out += f"Confidence interval size: {2.0*s*s90}\n"
            out += f"Confidence interval Delta: {s*s90}\n"
            out += f"Variation from noise is {values.SSE/values.SST*100.0}%. \n\n"
        out += f"The mean effect on {self.response_name} is {self.model.q[0]}.\n"
        out += self.state_confidence(0)
        out += '\n'
        for pidx in range(self.k):
            pname = self.pnames[pidx]
            plvl = self.plvl[pidx]
            out += f"When {pname} varies from {plvl[0]} to {plvl[1]}, the measured effect on {self.response_name} is {self.model.q[1<<pidx]}.\n"
            out += self.state_confidence(1<<pidx)
            out += self.state_variation(1<<pidx)
            out += self.draw_effect(1<<pidx)
            out += self.draw_variation(1<<pidx)
        out += '\n'
        # this is an inefficient loop, but it works.
        # TODO review the range()s and make sure <= is handled
        for nmix in range(2, self.k+1):
            for mix in range(1, N_to_the_k):
                if popcount(mix) == nmix:
                    out += f"The effect of interaction between "
                    for pidx in range(self.k):
                        if get_bit(mix, pidx) == 1:
                            out += f"{self.pnames[pidx]}, "
                    out += f"on {self.response_name} is {self.model.q[mix]}.\n"
                    out += self.state_confidence(mix)
                    out += self.state_variation(mix)
                    out += self.draw_effect(mix)
                    out += self.draw_variation(mix)
            out += '\n'
        out += ".............................Data..................................\n"
        out += "theta\n"
        out += ','.join([f"{self.model.q[khotidx]}" for khotidx in range(1,N_to_the_k)])
        out += '\nPct-Variation\n'
        out += ','.join([f"{self.variation(khotidx)}" for khotidx in range(1,N_to_the_k)])
        return out


    def get_string(self, khot):
        """
        The khot integer in binary, as a series of 1's and 0's
        """
        out = ''
        pidx = self.k-1
        while pidx >= 0:
            out += str(get_bit(khot, pidx))
            pidx -= 1
        return out


    def sign(self, i, j):
        out = popcount(j) + 1
        for b in range(self.k):
            out += get_bit(i, b)*get_bit(j, b)
        return -1 if out % 2 == 0 else 1


    def sum_SSnz(self):
        out = 0.0
        for i in range(1, self.N_to_the_k):
            out += self.values.SS[i]
        return out

