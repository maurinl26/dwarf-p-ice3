
source .venv/bin/activate
export PYTHONPATH=$HOME/maurinl26/dwarf-ice3-gt4py/src:$PYTHONPATH
echo $PYTHONPATH

echo "Test compile stencils"
python tests/main.py test-compile-stencils gt:cpu_kfirst | grep Compilation >> compilation.log
echo "Stencil compilation completed"

echo "Test components"
python tests/main.py test-components gt:cpu_kfirst
