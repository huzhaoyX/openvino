import numpy as np
from save_model import saveModel
import paddle as pdpd
import sys


def lod_array_length(name: str, x):
    pdpd.enable_static()
    tmp = pdpd.fluid.layers.zeros(shape=[10], dtype='int32')

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=x.dtype)
        i = pdpd.full(shape=[1], fill_value=9, dtype='int64' )
        arr = pdpd.fluid.layers.array_write(node_x, i=i)
        out = pdpd.fluid.layers.array_length(arr)

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
    data = np.array([[[[-1.3, 0.1, 1.3]], [[2.6, 3.7, 4.8]]]]).astype(np.float32)
    lod_array_length("lod_array_length", data)


if __name__ == "__main__":
    main()
