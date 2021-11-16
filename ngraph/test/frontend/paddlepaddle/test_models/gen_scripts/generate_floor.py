import numpy as np
from save_model import saveModel
import paddle as pdpd
import sys


def floor(name: str, x):
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=x.dtype)
        out = pdpd.floor(node_x)

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])

        saveModel(name, exe, feedkeys=['x'], fetchlist=[out],
                  inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    data_float32 = np.array([[[[-1.3, 0.1, 1.3]], [[2.6, 3.7, 4.8]]]]).astype(np.float32)
    data_float64 = np.array([[[[2.4, 0.8, 3.4]], [[3.6, 1.2, 4.9]]]]).astype(np.float64)
    floor("floor_float32", data_float32)
    floor("floor_float64", data_float64)


if __name__ == "__main__":
    main()
