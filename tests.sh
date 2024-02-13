
source .venv/bin/activate
export PYTHONPATH=$HOME/maurinl26/dwarf-ice3-gt4py/src:$PYTHONPATH
echo $PYTHONPATH

echo "Test compile stencils"
python tests/stencils/test_compile_stencils.py
echo "Stencil compilation completed"

echo "Test components"
echo "Test ice adjust"
python tests/components/test_ice_adjust.py
echo "Ice Adjust passed"
echo "Test aro adjust"
python tests/components/test_aro_adjust.py
echo "Aro Adjust passed"
echo "Components tests completed"
