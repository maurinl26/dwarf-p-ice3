import dace
import numpy as np

I = dace.symbol("I")
J = dace.symbol("J")
K = dace.symbol("K")

def add(
        a: dace.float32[I, J, K],
        b: dace.float32[I, J, K]
) -> dace.float32[I, J, K]:
    c = a + b
    return c

if __name__ == "__main__":

    sdfg = dace.program(add).to_sdfg()
    csdfg = sdfg.compile()


    a = np.ones((5, 5, 1), dtype=np.float32)
    b = np.ones((5, 5, 1), dtype=np.float32)
    c = np.zeros((5, 5, 1), dtype=np.float32)

    csdfg(a=a, b=b, c=c, I=5, J=5, K=1)

    print(np.sum(c) / (5 * 5 * 1))