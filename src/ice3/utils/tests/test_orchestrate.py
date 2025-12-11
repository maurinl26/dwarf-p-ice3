# -*- coding: utf-8 -*-
import pytest

from ice3.utils.orchestrate import orchestrate


@pytest.fixture(name="ice_adjust_component")
def ice_adjust_component_fixture():

    from ice3.components.ice_adjust import IceAdjust
    return IceAdjust


def test_orchestrate_component(ice_adjust_component):

    orchestrated_component = orchestrate(ice_adjust_component)

