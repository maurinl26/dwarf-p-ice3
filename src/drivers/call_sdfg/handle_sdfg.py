import dace

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