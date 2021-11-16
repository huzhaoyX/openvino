import numpy as np
from save_model import saveModel
import paddle as pdpd
import sys


def where(name: str, condition, x, y,):
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=x.dtype)
        node_y = pdpd.static.data(name='y', shape=y.shape, dtype=y.dtype)
        condition_temp = pdpd.static.data(name='condition', shape=condition.shape, dtype=condition.dtype)
        out = pdpd.where(condition_temp, node_x, node_y)

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
                feed={'x': x, 'y': y, 'condition': condition},
            fetch_list=[out])

        saveModel(name, exe, feedkeys=['x', 'y', 'condition'], fetchlist=[out],
                  inputs=[x, y, condition], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    data_x = np.array([[[[-1, 0, 1]], [[2, 3, 4]]]]).astype(np.float32)
    data_y = np.array([[[[2, 0, 3]], [[3, 1, 4]]]]).astype(np.float32)
    condition = np.array([[[[1, 0, 1]], [[0, 1, 1]]]]).astype(np.bool)

    where("where", condition, data_x, data_y)

if __name__ == "__main__":
    main()
