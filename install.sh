echo "Check python version"
python -c "import sys; print(sys.version)"

echo "Loading PHYEX as a submodule"
git submodule update --init --recursive

echo "Creating environment"
python3 -m venv .venv

echo "Installing requirements"
source .venv/bin/activate
pip install -r requirements.txt
