





from cpcrr.functions import liftsurf_4d_norm

from cpcrr.experiment import Twokr

import numpy as np

from cpcrr.methods import KrigingFitter



target = liftsurf_4d_norm

delta = 0.1

# number of sample point to build surrogate model
N = 18

rng = np.random.default_rng()
Sw, tc, Nz, Wdg = tuple(rng.random(4))

# Sw=0.5
# tc=0.5
# Nz=0.5
# Wdg=0.5


if __name__ == '__main__':

    Swmin, Swmax = Sw-delta, Sw+delta
    tcmin, tcmax = tc-delta, tc+delta
    Nzmin, Nzmax = Nz-delta, Nz+delta
    Wdgmin, Wdgmax = Wdg-delta, Wdg+delta
    Sw_sample = np.random.uniform(Swmin, Swmax, N)
    tc_sample = np.random.uniform(tcmin, tcmax, N)
    Nz_sample = np.random.uniform(Nzmin, Nzmax, N)
    Wdg_sample = np.random.uniform(Wdgmin, Wdgmax, N)
    datax = np.column_stack((Sw_sample,  tc_sample, Nz_sample, Wdg_sample))
    datay = target(Sw_sample, tc_sample, Nz_sample, Wdg_sample)
    model = KrigingFitter(
        xmin=(0,0),
        xmax=(1,1),
        datax=datax,
        datay=datay,
    )
    model.fit(
        # verbose=True,
    )


    # reference

    specification = ""
    specification += f'Sw, {Sw-delta}, {Sw+delta}\n'
    specification += f'tc, {tc-delta}, {tc+delta}\n'
    specification += f'Nz, {Nz-delta}, {Nz+delta}\n'
    specification += f'Wdg, {Wdg-delta}, {Wdg+delta}\n'
    response = "W\n"
    response += f'{target(Sw-delta, tc-delta, Nz-delta, Wdg-delta)}\n'
    response += f'{target(Sw-delta, tc-delta, Nz-delta, Wdg+delta)}\n'
    response += f'{target(Sw-delta, tc-delta, Nz+delta, Wdg-delta)}\n'
    response += f'{target(Sw-delta, tc-delta, Nz+delta, Wdg+delta)}\n'
    response += f'{target(Sw-delta, tc+delta, Nz-delta, Wdg-delta)}\n'
    response += f'{target(Sw-delta, tc+delta, Nz-delta, Wdg+delta)}\n'
    response += f'{target(Sw-delta, tc+delta, Nz+delta, Wdg-delta)}\n'
    response += f'{target(Sw-delta, tc+delta, Nz+delta, Wdg+delta)}\n'
    response += f'{target(Sw+delta, tc-delta, Nz-delta, Wdg-delta)}\n'
    response += f'{target(Sw+delta, tc-delta, Nz-delta, Wdg+delta)}\n'
    response += f'{target(Sw+delta, tc-delta, Nz+delta, Wdg-delta)}\n'
    response += f'{target(Sw+delta, tc-delta, Nz+delta, Wdg+delta)}\n'
    response += f'{target(Sw+delta, tc+delta, Nz-delta, Wdg-delta)}\n'
    response += f'{target(Sw+delta, tc+delta, Nz-delta, Wdg+delta)}\n'
    response += f'{target(Sw+delta, tc+delta, Nz+delta, Wdg-delta)}\n'
    response += f'{target(Sw+delta, tc+delta, Nz+delta, Wdg+delta)}\n'
    twokr = Twokr(
        specification=specification,
        # verbose=True,
    )
    print(f"Sw {Sw} tc {tc} Nz {Nz} Wdg {Wdg}")
    twokr.start(response=response)
    # todo multiple runs: logs are distinguishable



    # model


    specification = ""
    specification += f'Sw, {Sw-delta}, {Sw+delta}\n'
    specification += f'tc, {tc-delta}, {tc+delta}\n'
    specification += f'Nz, {Nz-delta}, {Nz+delta}\n'
    specification += f'Wdg, {Wdg-delta}, {Wdg+delta}\n'
    response = "W\n"
    response += f'{model(Sw-delta, tc-delta, Nz-delta, Wdg-delta)}\n'
    response += f'{model(Sw-delta, tc-delta, Nz-delta, Wdg+delta)}\n'
    response += f'{model(Sw-delta, tc-delta, Nz+delta, Wdg-delta)}\n'
    response += f'{model(Sw-delta, tc-delta, Nz+delta, Wdg+delta)}\n'
    response += f'{model(Sw-delta, tc+delta, Nz-delta, Wdg-delta)}\n'
    response += f'{model(Sw-delta, tc+delta, Nz-delta, Wdg+delta)}\n'
    response += f'{model(Sw-delta, tc+delta, Nz+delta, Wdg-delta)}\n'
    response += f'{model(Sw-delta, tc+delta, Nz+delta, Wdg+delta)}\n'
    response += f'{model(Sw+delta, tc-delta, Nz-delta, Wdg-delta)}\n'
    response += f'{model(Sw+delta, tc-delta, Nz-delta, Wdg+delta)}\n'
    response += f'{model(Sw+delta, tc-delta, Nz+delta, Wdg-delta)}\n'
    response += f'{model(Sw+delta, tc-delta, Nz+delta, Wdg+delta)}\n'
    response += f'{model(Sw+delta, tc+delta, Nz-delta, Wdg-delta)}\n'
    response += f'{model(Sw+delta, tc+delta, Nz-delta, Wdg+delta)}\n'
    response += f'{model(Sw+delta, tc+delta, Nz+delta, Wdg-delta)}\n'
    response += f'{model(Sw+delta, tc+delta, Nz+delta, Wdg+delta)}\n'
    twokr = Twokr(
        specification=specification,
        # verbose=True,
    )
    print(f"N {N}")
    print(f"Sw {Sw} tc {tc} Nz {Nz} Wdg {Wdg}")
    twokr.start(response=response)
    # todo multiple runs: logs are distinguishable
    print(f"Random center {Sw}, {tc}, {Nz}, {Wdg}")
    print(f"N-sample {N}")



