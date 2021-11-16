import numpy as np
from save_model import saveModel
import paddle as pdpd
import sys


def gather(name: str, x, index, axis):
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=x.dtype)
        index_1 = pdpd.static.data(name='index', shape=index.shape, dtype=index.dtype)
        out = pdpd.gather(x=node_x, index=index_1, axis=axis)

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
                feed={'x': x, 'index': index},
            fetch_list=[out])

        saveModel(name, exe, feedkeys=['x', 'index'], fetchlist=[out],
                  inputs=[x, index], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    input = np.array([[1,2],[3,4],[5,6]]).astype("int32")
    index = np.array([0,1]).astype("int32")
    gather("gather_test", input, index, axis=0)

if __name__ == "__main__":
    main()
