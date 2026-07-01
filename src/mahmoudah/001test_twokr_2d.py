

from cpcrr.functions import onevar, branin


from cpcrr.experiment import Twokr

target = branin
# target = onevar

delta = 0.1

xs = [
    # 0.0,
    0.1,
    # 0.2,
    # 0.3,
    # 0.4,
    # 0.5,
    # 0.6,
    # 0.7,
    # 0.8,
    # 0.9,
]
ys = [
    # 0.0,
    0.1,
    # 0.2,
    # 0.3,
    # 0.4,
    # 0.5,
    # 0.6,
    # 0.7,
    # 0.8,
    # 0.9,
]



if __name__ == '__main__':

    for x in xs:
        for y in ys:
            specification = ""
            response = "z\n"
            specification += f'x, {x-delta}, {x+delta}\n'
            specification += f'y, {y-delta}, {y+delta}\n'
            response += f'{branin(x-delta, y-delta)}\n'
            response += f'{branin(x-delta, y+delta)}\n'
            response += f'{branin(x+delta, y-delta)}\n'
            response += f'{branin(x+delta, y+delta)}\n'
            twokr = Twokr(
                specification=specification,
                # verbose=True,
            )
            print(f"x {x} y {y}")
            twokr.start(response=response)
            # todo multiple runs: logs are distinguishable









