from ice3_gt4py.components.component_to_sdfg import TestComponent

def test_component_to_sdfg(computational_grid, gt4py_config, phyex):

    gt4py_config.backend = "dace:cpu"

    component = TestComponent(
        gt4py_config=gt4py_config,
        computational_grid=computational_grid,
        phyex=phyex
    )

    import dace

    sdfg = dace.method(TestComponent.array_call)

