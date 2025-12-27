from spack.package import *

class DwarfPIce3(PythonPackage):
    """Description courte de votre projet scientifique."""

    homepage = "https://github.com/maurinl26/dwarf-p-ice3"
    pypi     = "dwarf-p-ice3/dwarf-p-ice3-1.0.0.tar.gz"
    git      = "https://github.com/maurinl26/dwarf-p-ice3.git"

    version("master", branch="master")
    version("1.0.0", sha256="VOTRE_SHA_ICI")

    # Variantes pour votre matrice
    variant("cuda", default=False, description="Build with CUDA/OpenACC support")
    variant("rocm", default=False, description="Build with ROCm support")
    variant("jax",  default=True,  description="Enable JAX backend")

    # Dépendances
    depends_on("python@3.10:", type=("build", "run"))
    depends_on("py-setuptools", type="build")
    depends_on("py-cython", type="build")
    depends_on("py-numpy", type=("build", "run"))
    
    depends_on("py-jax+cuda", when="+jax +cuda")
    depends_on("py-jax+rocm", when="+jax +rocm")
    depends_on("nvhpc", when="+cuda") # Pour OpenACC

    def setup_build_environment(self, env):
        if "+cuda" in self.spec:
            env.set("BUILD_TARGET", "gpu-acc")
        else:
            env.set("BUILD_TARGET", "cpu")