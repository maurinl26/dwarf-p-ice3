# -*- coding: utf-8 -*-
import dace

def orchestrate(component, backend = "dace:cpu"):

    # Init component with a DaCe backend
    component_dace = component(backend=backend)

    # Add decorated call
    orchestrated_call = dace.method(component_dace.__call__)
    setattr(component_dace, "orchestrated_call", orchestrated_call)

    # Return class with new method
    return component_dace

