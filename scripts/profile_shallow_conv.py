#!/usr/bin/env python
"""Profile shallow_convection to identify bottlenecks."""
import time
import numpy as np
import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from ice3.jax.convection.shallow_convection import shallow_convection, ConvectionParameters

# Create test data (small domain)
nlon = 50
nkt = 30
ikb = 2
ike = nkt - 1

# Simple atmospheric profiles
ptt = np.linspace(300, 250, nkt, dtype=np.float32)
ptt = np.tile(ptt, (nlon, 1))
ppabst = np.linspace(100000, 20000, nkt, dtype=np.float32)
ppabst = np.tile(ppabst, (nlon, 1))
prhodref = np.linspace(1.2, 0.3, nkt, dtype=np.float32)
prhodref = np.tile(prhodref, (nlon, 1))
pzz = np.linspace(0, 10000, nkt, dtype=np.float32)
pzz = np.tile(pzz, (nlon, 1))

# Moisture variables
prwt = np.full((nlon, nkt), 0.005, dtype=np.float32)
prct = np.full((nlon, nkt), 0.0005, dtype=np.float32)
prit = np.full((nlon, nkt), 0.0001, dtype=np.float32)
pnet = np.zeros((nlon, nkt), dtype=np.float32)

# Tendencies
pthlt = np.zeros((nlon, nkt), dtype=np.float32)
prwt_tend = np.zeros((nlon, nkt), dtype=np.float32)
prct_tend = np.zeros((nlon, nkt), dtype=np.float32)
prit_tend = np.zeros((nlon, nkt), dtype=np.float32)
pnet_tend = np.zeros((nlon, nkt), dtype=np.float32)

# Vertical velocity
pw = np.full((nlon, nkt), 0.1, dtype=np.float32)

# Convert to JAX arrays
jax_data = {
    'ptt': jnp.asarray(ptt),
    'ppabst': jnp.asarray(ppabst),
    'prhodref': jnp.asarray(prhodref),
    'pzz': jnp.asarray(pzz),
    'prwt': jnp.asarray(prwt),
    'prct': jnp.asarray(prct),
    'prit': jnp.asarray(prit),
    'pnet': jnp.asarray(pnet),
    'pthlt': jnp.asarray(pthlt),
    'prwt_tend': jnp.asarray(prwt_tend),
    'prct_tend': jnp.asarray(prct_tend),
    'prit_tend': jnp.asarray(prit_tend),
    'pnet_tend': jnp.asarray(pnet_tend),
    'pw': jnp.asarray(pw),
    'kch1': 1,
}

params = ConvectionParameters()

# Warm up JIT
print("Warming up JIT...")
for _ in range(3):
    result = shallow_convection(**jax_data, convection_params=params)
jax.block_until_ready(result)
print("JIT warm-up complete\n")

# Now profile
print("Profiling shallow_convection...")
n_runs = 100
times = []

for i in range(n_runs):
    start = time.perf_counter()
    result = shallow_convection(**jax_data, convection_params=params)
    jax.block_until_ready(result)
    elapsed = time.perf_counter() - start
    times.append(elapsed)

times = np.array(times)
print(f"\nResults from {n_runs} runs:")
print(f"Mean time: {times.mean()*1000:.3f} ms")
print(f"Std dev: {times.std()*1000:.3f} ms")
print(f"Min time: {times.min()*1000:.3f} ms")
print(f"Max time: {times.max()*1000:.3f} ms")
print(f"Median time: {np.median(times)*1000:.3f} ms")

# Calculate throughput
total_points = nlon * nkt
throughput = total_points / times.mean()
print(f"\nThroughput: {throughput/1e3:.2f} K points/s")
print(f"Domain size: {nlon} × {nkt} = {total_points:,} points")
