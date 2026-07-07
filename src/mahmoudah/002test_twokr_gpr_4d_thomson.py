



from queueg.experiment import Twokr

from cpcrr.methods import KrigingFitter

import numpy as np




def fn1(x, y, z, w):
    return 0*x + 0*y + -0.3772684207887429*y*x + 0*z + 0.5225621353568051*z*x + 0.5774598217196054*z*y + -0.7007822003600315*x*y*z + 0*w + 0.6443523721811273*w*x + -0.9171332532250895*w*y + -0.5111410589542336*w*y*x +-0.9440057897312653*w*z + -0.0682772319767766*w*z*x + -0.5800488898115175*w*z*y + 0.31782667632356065*w*z*y*x

def fn2(x, y, z, w):
    return 0*x + 0*y + -0.1611893480167601*y*x + 0*z + 0.7653665895258082*z*x + 0.7382092671110967*z*y + -0.26029194642164755*x*y*z + 0*w + 0.1545314158770339*w*x + -0.824568640457799*w*y + 0.8261809792702959*w*y*x +0.9639405802947414*w*z + 0.09693205157938145*w*z*x + 0.5138745496000947*w*z*y + -0.6030300522088818*w*z*y*x

def fn3(x, y, z, w):
    return  -60.30300522088818*(w*z*y*x)**16

def fn4(x, y, z, w):
    return  np.cos(w*z*y*x)**4

def fn5(x,y,z,w):
    return (w*z*y*x)**8



N = 32

if N == 4:
    datax = np.array([
        [ 1.00000000e+00,  7.63278329e-17, -5.55111512e-17,  4.16333634e-17],
        [-3.33168064e-01,  6.58130144e-01,  8.73300501e-02,  6.69505203e-01],
        [-3.33507675e-01,  6.54383541e-02, -6.91206365e-01, -6.37749334e-01],
        [-3.33227401e-01, -7.23584970e-01,  6.03648499e-01, -3.15084107e-02],
    ])
elif N == 8:
    datax = np.array([
        [ 1.00000000e+00,  2.77555756e-17, -3.46944695e-18, -5.55111512e-17],
        [ 1.08274976e-04, -1.42813100e-01,  9.89317422e-01, -2.92479837e-02],
        [ 4.02777203e-05,  5.18122547e-01,  4.90969771e-02, -8.53896078e-01],
        [-2.33660057e-04, -5.18014430e-01, -4.96331674e-02,  8.53930644e-01],
        [-9.99999952e-01, -6.86341237e-05,  2.86480409e-04,  9.68103797e-05],
        [-6.73564950e-05,  8.43395935e-01,  1.37175740e-01,  5.19486390e-01],
        [ 2.01835660e-04,  1.42592250e-01, -9.89349424e-01,  2.92425642e-02],
        [ 2.51621066e-04, -8.43289220e-01, -1.36940122e-01, -5.19721686e-01],
    ])
elif N == 16:
    datax = np.array([
        [ 1.00000000e+00, -5.55111512e-17,  5.55111512e-17,  1.66533454e-16],
        [ 4.37088692e-01, -1.01719810e-01,  8.28628298e-01, -3.34636667e-01],
        [-6.33459146e-01,  4.72885094e-01,  2.25760340e-01,  5.69334231e-01],
        [-6.34252527e-01, -6.01768483e-01,  6.02375883e-02,  4.81632493e-01],
        [-6.34238896e-01,  6.14764628e-01, -1.63181292e-01, -4.39519443e-01],
        [ 1.91595590e-01, -1.61574673e-01,  6.83606302e-01,  6.85468583e-01],
        [-4.37891949e-01,  1.00388457e-01, -8.28532491e-01,  3.34225538e-01],
        [ 3.45981491e-01,  6.68819939e-01, -6.03700411e-01, -2.61768049e-01],
        [-6.17586852e-01, -8.83904411e-02,  7.24786544e-01, -2.92332134e-01],
        [ 3.47607480e-01, -4.05549798e-01, -7.69870328e-01, -3.49282235e-01],
        [ 3.47146346e-01,  5.25784635e-01, -2.14114359e-01,  7.46454937e-01],
        [-6.33590750e-01, -4.59670393e-01, -3.30408981e-01, -5.27347890e-01],
        [ 1.91072477e-01, -9.30129020e-01,  2.24732601e-01, -2.18738594e-01],
        [ 1.91868878e-01,  8.41409048e-01,  4.99711619e-01, -7.41986960e-02],
        [ 1.91470883e-01,  7.37624346e-02,  4.06736677e-02, -9.77877117e-01],
        [ 3.46276331e-01, -5.48654734e-01, -3.79910733e-01,  6.59347041e-01],
    ])
elif N == 32:
    datax = np.array([
        [ 1.00000000e+00, -1.04083409e-17,  2.22044605e-16,  1.38777878e-16],
        [ 5.68233924e-01,  2.16133978e-01, -7.83445311e-01,  1.28878841e-01],
        [-4.68144535e-01,  8.14100268e-01,  3.14124239e-01, -1.39310483e-01],
        [-1.71159856e-01, -5.73212072e-01,  6.49231154e-01, -4.69713884e-01],
        [-7.87188377e-01,  4.61186567e-02,  3.84661415e-01, -4.79836561e-01],
        [ 5.94928832e-01, -8.73086166e-02,  7.54517904e-01, -2.62944143e-01],
        [-8.40783735e-01,  3.62270855e-01, -2.90899659e-01,  2.77884737e-01],
        [-3.49553654e-01, -1.79981354e-01,  3.84829790e-02,  9.18660990e-01],
        [-1.37529085e-01, -8.86867764e-01, -2.03556969e-01, -3.91300244e-01],
        [-7.84167841e-02, -8.69976929e-01, -8.66642529e-02,  4.79040978e-01],
        [ 4.51202708e-01, -1.23964270e-02,  6.13958132e-01,  6.47547571e-01],
        [-8.00237866e-01, -5.85576123e-01,  6.27176730e-02,  1.13077207e-01],
        [ 5.64236836e-01, -7.56559999e-01,  3.21558487e-01,  7.65107877e-02],
        [-2.69014569e-01,  3.67204437e-01, -8.88216701e-01, -6.21542778e-02],
        [-2.98090499e-01, -4.16948000e-01, -7.82804625e-01,  3.52864476e-01],
        [-7.51951565e-01,  9.84362728e-02,  5.24407478e-01,  3.87138142e-01],
        [ 6.24731129e-01,  5.32309046e-01, -6.95347443e-02,  5.67029995e-01],
        [ 5.10190822e-01, -3.03003742e-01, -2.41382594e-01,  7.67872712e-01],
        [-1.39189715e-01,  6.30972625e-01,  3.48887078e-01,  6.78805994e-01],
        [ 8.96606789e-02,  3.75428051e-01,  4.87464154e-01, -7.83194382e-01],
        [-3.50058211e-01,  5.69229905e-01, -3.02536554e-01, -6.79638284e-01],
        [ 1.55664801e-01, -1.56448060e-01, -7.60299782e-01, -6.10931023e-01],
        [-1.10165290e-01,  3.44393791e-01, -5.51925186e-01,  7.51422062e-01],
        [-2.21768263e-01, -2.29258906e-01, -2.34873111e-02, -9.47474294e-01],
        [ 4.88591014e-01,  7.35273696e-01,  4.64543694e-01, -6.96460347e-02],
        [-7.37156513e-01, -2.22627834e-01, -5.18087587e-01, -3.72320259e-01],
        [ 5.73331472e-01,  5.09333680e-01, -2.62818151e-01, -5.85488553e-01],
        [-1.61148456e-01, -5.38551439e-01,  7.33195579e-01,  3.82645745e-01],
        [ 6.36082776e-02,  9.41567748e-01, -3.25335587e-01,  5.96734367e-02],
        [ 5.25712315e-01, -6.14770955e-01, -5.87938607e-01, -3.38074578e-03],
        [ 6.00945274e-01, -3.59597670e-01,  2.39483731e-02, -7.13428881e-01],
        [-1.37400479e-01,  2.51159387e-01,  9.57940832e-01, -1.97340691e-02],
    ])


rng = np.random.default_rng()


delta = 0.1


# x, y, z, w = tuple(rng.random(4))
x, y, z, w = 1,1,1,1

center = np.array([x,y,z,w])

if __name__ == '__main__':


    target = fn5

    xmin, xmax = x-delta, x+delta
    ymin, ymax = y-delta, y+delta
    zmin, zmax = z-delta, z+delta
    wmin, wmax = w-delta, w+delta
    # transform and scale unit sphere
    datax = np.array(center) + delta*datax
    datay = target(datax[:,0], datax[:,1], datax[:,2], datax[:,3])
    model = KrigingFitter(
        xmin=(xmin, ymin, zmin, wmin),
        xmax=(xmax, ymax, zmax, wmax),
        datax=datax,
        datay=datay,
    )
    model.fit(
        # verbose=True,
    )


    # reference

    specification = ""
    specification += f'x, {x-delta}, {x+delta}\n'
    specification += f'y, {y-delta}, {y+delta}\n'
    specification += f'z, {z-delta}, {z+delta}\n'
    specification += f'w, {w-delta}, {w+delta}\n'
    response = "f\n"
    response += f'{target(x-delta, y-delta, z-delta, w-delta)}\n'
    response += f'{target(x-delta, y-delta, z-delta, w+delta)}\n'
    response += f'{target(x-delta, y-delta, z+delta, w-delta)}\n'
    response += f'{target(x-delta, y-delta, z+delta, w+delta)}\n'
    response += f'{target(x-delta, y+delta, z-delta, w-delta)}\n'
    response += f'{target(x-delta, y+delta, z-delta, w+delta)}\n'
    response += f'{target(x-delta, y+delta, z+delta, w-delta)}\n'
    response += f'{target(x-delta, y+delta, z+delta, w+delta)}\n'
    response += f'{target(x+delta, y-delta, z-delta, w-delta)}\n'
    response += f'{target(x+delta, y-delta, z-delta, w+delta)}\n'
    response += f'{target(x+delta, y-delta, z+delta, w-delta)}\n'
    response += f'{target(x+delta, y-delta, z+delta, w+delta)}\n'
    response += f'{target(x+delta, y+delta, z-delta, w-delta)}\n'
    response += f'{target(x+delta, y+delta, z-delta, w+delta)}\n'
    response += f'{target(x+delta, y+delta, z+delta, w-delta)}\n'
    response += f'{target(x+delta, y+delta, z+delta, w+delta)}\n'
    twokr = Twokr(
        specification=specification,
        # verbose=True,
    )
    print(f"x {x} y {y} z {z} w {w}")
    twokr.start(response=response)
    # todo multiple runs: logs are distinguishable


    reference_effects = np.array([twokr.effect(idx) for idx in range(1,16)])
    reference_variations = np.array([twokr.variation(idx) for idx in range(1,16)])
    data_reference = np.hstack((reference_effects, reference_variations))
    reference_sum_variations = np.sum(reference_variations)


    # model


    specification = ""
    specification += f'x, {x-delta}, {x+delta}\n'
    specification += f'y, {y-delta}, {y+delta}\n'
    specification += f'z, {z-delta}, {z+delta}\n'
    specification += f'w, {w-delta}, {w+delta}\n'
    response = "f\n"
    response += f'{model(x-delta, y-delta, z-delta, w-delta)}\n'
    response += f'{model(x-delta, y-delta, z-delta, w+delta)}\n'
    response += f'{model(x-delta, y-delta, z+delta, w-delta)}\n'
    response += f'{model(x-delta, y-delta, z+delta, w+delta)}\n'
    response += f'{model(x-delta, y+delta, z-delta, w-delta)}\n'
    response += f'{model(x-delta, y+delta, z-delta, w+delta)}\n'
    response += f'{model(x-delta, y+delta, z+delta, w-delta)}\n'
    response += f'{model(x-delta, y+delta, z+delta, w+delta)}\n'
    response += f'{model(x+delta, y-delta, z-delta, w-delta)}\n'
    response += f'{model(x+delta, y-delta, z-delta, w+delta)}\n'
    response += f'{model(x+delta, y-delta, z+delta, w-delta)}\n'
    response += f'{model(x+delta, y-delta, z+delta, w+delta)}\n'
    response += f'{model(x+delta, y+delta, z-delta, w-delta)}\n'
    response += f'{model(x+delta, y+delta, z-delta, w+delta)}\n'
    response += f'{model(x+delta, y+delta, z+delta, w-delta)}\n'
    response += f'{model(x+delta, y+delta, z+delta, w+delta)}\n'
    twokr = Twokr(
        specification=specification,
        # verbose=True,
    )
    print(f"N {N}")
    print(f"x {x} y {y} z {z} w {w}")
    twokr.start(response=response)
    # todo multiple runs: logs are distinguishable
    print(f"Random center {x}, {y}, {z}, {w}")
    print(f"N-sample {N}")



    model_effects = np.array([twokr.effect(idx) for idx in range(1,16)])
    model_variations = np.array([twokr.variation(idx) for idx in range(1,16)])
    data_model = np.hstack((model_effects, model_variations))
    model_sum_variations = np.sum(model_variations)

    print(f"model_sum_variations {model_sum_variations}")
    print(f"reference_sum_variations {reference_sum_variations}")

    effects_errors = np.abs(model_effects - reference_effects)/(np.maximum(np.ones_like(reference_effects), np.abs(reference_effects)))*100.0
    variations_errors = np.abs(np.abs(model_variations) - np.abs(reference_variations))/(np.maximum(np.ones_like(reference_variations), np.abs(reference_variations)))*100.0

    print(f"reference_effects {[f'{reference_effects[i]:.02}' for i in range(effects_errors.shape[0])]}")
    print(f"model_effects {[f'{model_effects[i]:.02}' for i in range(effects_errors.shape[0])]}")
    print(f"effects_errors {[f'{effects_errors[i]:.02}' for i in range(effects_errors.shape[0])]}")
    print(f"variations_errors {[f'{variations_errors[i]:.02}' for i in range(variations_errors.shape[0])]}")


    # data_reference = np.concat(np.array([twokr.effect(idx) for idx in range(1,16)]), np.array([twokr.variation(idx) for idx in range(1,16)]))

