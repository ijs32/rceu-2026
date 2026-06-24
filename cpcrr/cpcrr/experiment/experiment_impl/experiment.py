

from os.path import (
    join as os_path_join,
)

from physixlib.visutil import (
    Figure,
)



class Experiment:
    """
    Base class for an experiment,
    in the sense of statistical theory.

    """


    def __init__(
        self,
        specification = None,
        verbose = False,
    ):
        self.location = None
        self.fig = Figure()
        if specification[-4:] != '.csv':
            self.specification = specification.strip()
        else:
            with open(specification, 'r') as f:
                self.specification = f.read().strip()
        lines = self.specification.split('\n')
        self.k = len(lines)
        self.N = len(lines[0].split(',')) - 1
        self.N_to_the_k = self.N**self.k
        self.pnames = []
        self.plvl = []
        self.response_name = None
        # response 'raw' data
        self.data = None
        # organized 'output' data, can vary between experiments
        self.X = None
        self._log = ""
        self.v = verbose
        # > read specification
        lines = self.specification.split('\n')
        for line in lines:
            items = line.split(',')
            # > write name
            self.pnames.append(items[0].strip())
            levels = []
            for leveli in items[1:]:
                levels.append(leveli.strip())
            # > write level list
            self.plvl.append(levels)
        self.log(str(self.plvl))
        self.log(str(self.pnames))
        # the (maximum) number of repetitions in levels/groups
        self.r = None
        # the numbers of repetitions in levels/groups, if this is allowed to vary
        self.rs = None
        # the experiment samples (levels and repetitions/trials)
        self.esmpls = None
        # total degrees of freedom (for the entire experiment)
        self.DF = None


    class ExperimentSample:

        def __init__(
                self,
                r,
        ):
            self.response = 0.0
            self.raw_data = []
            for _ in range(r):
                self.raw_data.append(None)

        def r(self):
            return len([x for x in self.raw_data if x is not None])


    def _init(self):
        lines = self.data.split('\n')
        # > find rs, r, DF
        self.rs = []
        r = 0
        DF = 0
        for line in lines[1:]:
            r_ = len(line.split(','))
            self.rs.append(r_)
            r = r_ if r_ > r else r
            DF += r_
        self.DF = DF
        self.r = r
        self.esmpls = []
        for _ in range(self.N_to_the_k):
            self.esmpls.append(self.ExperimentSample(r=r))


    def _read_data(self):
        """
        Read the individual responses into experiments
        and populate names and levels of parameters.
        """
        lines = self.data.split('\n')
        # We assume there is *always* a header line containing
        # the response name.
        self.response_name = lines[0].split(',')[0].strip()
        self.log(f"response: {self.response_name}")
        for i, line in enumerate(lines[1:]):
            smpl = self.esmpls[i]
            trials = line.split(',')
            for j, trial in enumerate(trials):
                trl = float(trial)
                smpl.raw_data[j] = trl
                smpl.response += trl
            smpl.response /= len(trials)
            self.log(f"expt {i}: {smpl.raw_data}")
            self.log(f"response {i}: {smpl.response}")



    def log(self, msg):
        if self.v:
            print(f"{msg}\n")
        self._log += msg


