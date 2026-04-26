"""
Launch a dwarf-p-ice3 pod on RunPod.

Usage:
    RUNPOD_API_KEY=<key> python scripts/runpod_launch.py [--task TASK]

TASK choices:
    smoke-cpu       JAX-only smoke tests — no GPU required    [default]
    smoke-gpu       SURFEX + PHYEX GPU bridge smoke tests
    test-phyex      PHYEX ice_adjust + rain_ice JAX standalone
    test-surfex     SURFEX GPU bridge full suite
    test-components All component tests
    test-physics    AromePhysics standalone end-to-end
    bench           Performance benchmarks → /workspace/bench.json
    shell           Interactive bash, no auto-run

Requirements:
    pip install runpod

Environment variables:
    RUNPOD_API_KEY          RunPod API key (runpod.io → Settings → API Keys)
    SSH_KEY_PATH            Path to public SSH key (default: ~/.ssh/id_ed25519.pub)
    RUNPOD_TEMPLATE_ID      Pod template ID with GHCR auth baked in (see --register-ghcr)
    RUNPOD_REGISTRY_AUTH_ID Existing registry credential ID (skip auth creation)
    GHCR_USERNAME           GitHub username for GHCR (used by --register-ghcr)
    GHCR_PAT                GitHub PAT with read:packages scope

Private GHCR image setup (one-time):
    1. Create a GitHub PAT: github.com → Settings → Developer settings → Tokens (classic)
       → New token → scope: read:packages
    2. Register with RunPod (creates registry auth + pod template):
       RUNPOD_API_KEY=<key> GHCR_USERNAME=maurinl26 GHCR_PAT=<pat> \\
       python scripts/runpod_launch.py --register-ghcr
    3. Copy the printed RUNPOD_TEMPLATE_ID into your .env file.

Workflow:
    # 1. Launch a pod and run smoke tests
    python scripts/runpod_launch.py --task smoke-gpu

    # 2. Check pod status / SSH info
    python scripts/runpod_launch.py --status <POD_ID>

    # 3. Sync local changes and re-run without rebuild
    python scripts/runpod_launch.py --sync <POD_ID> --task test-phyex

    # 4. Tail live logs
    python scripts/runpod_launch.py --logs <POD_ID> --task test-physics

    # 5. Stop the pod
    python scripts/runpod_launch.py --stop <POD_ID>
"""

import argparse
import datetime
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VALID_TASKS = [
    "smoke-cpu",
    "smoke-gpu",
    "test-phyex",
    "test-surfex",
    "test-components",
    "test-physics",
    "bench",
    "shell",
]

# Tasks that produce /workspace/bench.json (auto-fetched after run).
BENCH_TASKS = {"bench"}

IMAGE = os.environ.get(
    "ICE3_IMAGE",
    "ghcr.io/maurinl26/dwarf-p-ice3:runpod",
)
TEMPLATE_NAME  = "ice3-runpod-dev"
VOLUME_NAME    = "ice3-workspace"
VOLUME_MOUNT   = "/workspace"
CONTAINER_DISK_GB = 20
GPU_TYPE       = "NVIDIA A100 80GB SXM"

RSYNC_EXCLUDES = [
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    ".pytest_cache/",
    "*.egg-info/",
    "build/",
    ".venv/",
    "*.so",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _task_cmd(task: str) -> str:
    """SSH command to dispatch *task* on a running pod."""
    return (
        f"cd /opt/ice3 && "
        f"TASK={task} JAX_PLATFORM_NAME=cuda "
        f"bash /opt/ice3/container/runpod_startup.sh"
    )


def _log_file(task: str) -> str:
    return f"/workspace/{task.replace('-', '_')}.log"


def get_api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not key:
        sys.exit("ERROR: set RUNPOD_API_KEY  (runpod.io → Settings → API Keys)")
    return key


def get_pub_key(path: str) -> str:
    expanded = os.path.expanduser(path)
    if not os.path.exists(expanded):
        sys.exit(f"ERROR: SSH public key not found at {expanded}")
    return open(expanded).read().strip()


def _private_key(pub_key_path: str) -> str:
    path = os.path.expanduser(pub_key_path)
    return path[:-4] if path.endswith(".pub") else path


def _get_pod_ssh_endpoint(pod_id: str) -> tuple[str, int]:
    import runpod as rp
    rp.api_key = get_api_key()
    pod = rp.get_pod(pod_id)
    if not pod:
        sys.exit(f"Pod {pod_id} not found.")
    ports = (pod.get("runtime") or {}).get("ports") or []
    for port in ports:
        if port.get("privatePort") == 22 and port.get("isIpPublic"):
            return port.get("ip", "ssh.runpod.io"), int(port["publicPort"])
    sys.exit(
        f"Pod {pod_id}: SSH port not yet assigned — pod may still be starting.\n"
        f"  python scripts/runpod_launch.py --status {pod_id}"
    )


def _ssh_info(pod: dict) -> str:
    ports = (pod.get("runtime") or {}).get("ports") or []
    for port in ports:
        if port.get("privatePort") == 22 and port.get("isIpPublic"):
            return f"ssh root@{port.get('ip', 'ssh.runpod.io')} -p {port['publicPort']} -i ~/.ssh/id_ed25519"
    return "(SSH port not yet assigned)"


def get_network_volumes(api_key: str) -> list:
    from runpod.api.graphql import run_graphql_query
    import runpod
    runpod.api_key = api_key
    result = run_graphql_query(
        "{ myself { networkVolumes { id name size dataCenter { id } } } }"
    )
    return result.get("data", {}).get("myself", {}).get("networkVolumes", [])


def get_volume_id(api_key: str, name: str) -> str:
    volumes = get_network_volumes(api_key)
    for v in volumes:
        if v["name"] == name:
            return v["id"]
    available = [v["name"] for v in volumes]
    sys.exit(
        f"ERROR: network volume '{name}' not found.\n"
        f"  Available: {available}\n"
        f"  Create it at runpod.io → Storage → + Network Volume"
    )


def get_gpu_candidates(rp, display_name: str) -> list:
    gpus = rp.get_gpus()
    matches = [g for g in gpus if display_name.lower() in g["id"].lower()]
    if not matches:
        print("Available GPU types:")
        for g in gpus:
            print(f"  {g['id']}  ({g.get('memoryInGb', '?')} GB)")
        sys.exit(f"ERROR: no GPU matching '{display_name}'")
    return [g["id"] for g in sorted(matches, key=lambda g: g.get("memoryInGb", 0), reverse=True)]


# ---------------------------------------------------------------------------
# register-ghcr: one-time GHCR credentials + template setup
# ---------------------------------------------------------------------------

def _get_existing_template(name: str) -> dict | None:
    from runpod.api.graphql import run_graphql_query
    result = run_graphql_query(
        "{ myself { podTemplates { id name containerRegistryAuthId } } }"
    )
    templates = result.get("data", {}).get("myself", {}).get("podTemplates", [])
    for t in templates:
        if t["name"] == name:
            return t
    return None


def register_ghcr() -> None:
    """Register GHCR credentials with RunPod and create (or reuse) a pod template."""
    import runpod as rp

    rp.api_key = get_api_key()

    auth_id = os.environ.get("RUNPOD_REGISTRY_AUTH_ID", "")
    if auth_id:
        print(f"Reusing existing registry auth ID: {auth_id}")
    else:
        username = os.environ.get("GHCR_USERNAME", "")
        pat = os.environ.get("GHCR_PAT", "")
        if not username or not pat:
            sys.exit(
                "ERROR: set GHCR_USERNAME + GHCR_PAT (or RUNPOD_REGISTRY_AUTH_ID)\n"
                "  GHCR_PAT must be a GitHub PAT with read:packages scope."
            )
        print("Creating GHCR registry credential...")
        auth = rp.create_container_registry_auth("ghcr-ice3", username, pat)
        auth_id = auth["id"]
        print(f"  Registry auth ID : {auth_id}")
        print(f"  Add to .env      : RUNPOD_REGISTRY_AUTH_ID={auth_id}")

    existing = _get_existing_template(TEMPLATE_NAME)
    if existing:
        template_id = existing["id"]
        tpl_auth = existing.get("containerRegistryAuthId")
        if tpl_auth and tpl_auth != auth_id:
            print(f"WARNING: template '{TEMPLATE_NAME}' uses a different auth ({tpl_auth}).")
            print("         If image pulls fail, delete the template on runpod.io and re-run.")
        else:
            print(f"Template '{TEMPLATE_NAME}' already exists — reusing it.")
    else:
        print(f"Creating pod template '{TEMPLATE_NAME}'...")
        tpl = rp.create_template(
            name=TEMPLATE_NAME,
            image_name=IMAGE,
            container_disk_in_gb=CONTAINER_DISK_GB,
            ports="22/tcp",
            registry_auth_id=auth_id,
        )
        template_id = tpl["id"]

    print(f"  Template ID      : {template_id}")
    print()
    print("Add to your .env:")
    print(f"  RUNPOD_TEMPLATE_ID={template_id}")


# ---------------------------------------------------------------------------
# create_pod
# ---------------------------------------------------------------------------

def create_pod(task: str, ssh_key_path: str, gpu_name: str) -> None:
    import runpod as rp
    import runpod.error as rp_error

    api_key = get_api_key()
    rp.api_key = api_key
    pub_key = get_pub_key(ssh_key_path)
    template_id = os.environ.get("RUNPOD_TEMPLATE_ID") or None

    if template_id:
        print(f"Using template ID: {template_id}")
    else:
        print("WARNING: RUNPOD_TEMPLATE_ID not set — image pull may fail for private GHCR.")
        print("         Run: python scripts/runpod_launch.py --register-ghcr")

    # Network volume is optional — skip gracefully if not found.
    try:
        volume_id = get_volume_id(api_key, VOLUME_NAME)
    except SystemExit:
        print(f"WARNING: network volume '{VOLUME_NAME}' not found — pod will run without persistent storage.")
        volume_id = None

    gpu_candidates = get_gpu_candidates(rp, gpu_name)

    pod = None
    for gpu_type_id in gpu_candidates:
        print(f"Trying  task={task}  gpu={gpu_type_id}")
        try:
            kwargs = dict(
                name=f"ice3-{task}",
                image_name="" if template_id else IMAGE,
                gpu_type_id=gpu_type_id,
                gpu_count=1,
                container_disk_in_gb=CONTAINER_DISK_GB,
                ports="22/tcp",
                template_id=template_id,
                env={
                    "TASK": task,
                    "RUNPOD_PUBLIC_KEY": pub_key,
                },
            )
            if volume_id:
                kwargs["network_volume_id"] = volume_id
            pod = rp.create_pod(**kwargs)
            break
        except rp_error.QueryError as e:
            if "no longer any instances available" in str(e):
                print(f"  → {gpu_type_id} full, trying next...")
                continue
            raise

    if pod is None:
        sys.exit(f"ERROR: all {gpu_name} GPU instances are full. Retry later or try --gpu H100.")

    pod_id = pod["id"]
    print()
    print(f"Pod created")
    print(f"  ID     : {pod_id}")
    print(f"  Image  : {IMAGE}")
    print(f"  Task   : {task}")
    print()
    print("Next steps:")
    print(f"  Wait for SSH ready : python scripts/runpod_launch.py --wait {pod_id}")
    print(f"  Show SSH info      : python scripts/runpod_launch.py --status {pod_id}")
    print(f"  Tail logs          : python scripts/runpod_launch.py --logs {pod_id} --task {task}")
    print(f"  Stop pod           : python scripts/runpod_launch.py --stop {pod_id}")

    # Emit pod_id for GitHub Actions downstream steps.
    gha_output = os.environ.get("GITHUB_OUTPUT", "")
    if gha_output:
        with open(gha_output, "a") as f:
            f.write(f"pod_id={pod_id}\n")


# ---------------------------------------------------------------------------
# sync_pod: rsync Python-only changes without a full rebuild
# ---------------------------------------------------------------------------

def sync_pod(pod_id: str, ssh_key_path: str, task: str | None = None) -> None:
    """Rsync local source and tests to a running pod, then optionally run a task.

    Covers Python-only fixes. For Dockerfile or dependency changes, rebuild the image.

    Sync targets:
      src/                     → /opt/ice3/src/
      tests/                   → /opt/ice3/tests/
      container/runpod_startup.sh → /opt/ice3/container/runpod_startup.sh
    """
    host, port = _get_pod_ssh_endpoint(pod_id)
    priv_key = _private_key(ssh_key_path)
    ssh_e = f"ssh -p {port} -i {priv_key} -o StrictHostKeyChecking=no -o LogLevel=ERROR"

    print(f"Syncing to pod {pod_id}  ({host}:{port})")

    sync_targets = [
        ("src/",                           f"root@{host}:/opt/ice3/src/"),
        ("tests/",                         f"root@{host}:/opt/ice3/tests/"),
        ("container/runpod_startup.sh",    f"root@{host}:/opt/ice3/container/runpod_startup.sh"),
    ]

    for local, remote in sync_targets:
        print(f"  rsync {local} → {remote}")
        rsync_cmd = ["rsync", "-avz", "-e", ssh_e]
        for exc in RSYNC_EXCLUDES:
            rsync_cmd += ["--exclude", exc]
        rsync_cmd += [local, remote]
        result = subprocess.run(rsync_cmd)
        if result.returncode != 0:
            sys.exit(f"rsync failed for {local}")

    print("Sync complete.")

    if task is None:
        return
    if task == "shell":
        print("Hint: for interactive mode SSH in directly:")
        print(f"  ssh root@{host} -p {port} -i {priv_key}")
        return

    print(f"\nRunning task '{task}' on pod (output streamed live)...")
    subprocess.run([
        "ssh", "-p", str(port), "-i", priv_key,
        "-o", "StrictHostKeyChecking=no", "-o", "LogLevel=ERROR",
        f"root@{host}", _task_cmd(task),
    ])
    _maybe_fetch_bench(task, host, port, priv_key, ssh_e)


# ---------------------------------------------------------------------------
# logs: tail a task log file on a running pod
# ---------------------------------------------------------------------------

def logs_pod(pod_id: str, ssh_key_path: str, task: str) -> None:
    """Stream the log file for *task* from the running pod."""
    host, port = _get_pod_ssh_endpoint(pod_id)
    priv_key = _private_key(ssh_key_path)
    log = _log_file(task)
    print(f"Tailing {log} on pod {pod_id}  ({host}:{port})")
    print("(Ctrl-C to detach)\n")
    subprocess.run([
        "ssh", "-p", str(port), "-i", priv_key,
        "-o", "StrictHostKeyChecking=no", "-o", "LogLevel=ERROR",
        f"root@{host}", f"tail -f {log}",
    ])


# ---------------------------------------------------------------------------
# stop / list / status / wait
# ---------------------------------------------------------------------------

def stop_pod(pod_id: str) -> None:
    import runpod as rp
    rp.api_key = get_api_key()
    rp.terminate_pod(pod_id)
    print(f"Pod {pod_id} terminated.")


def list_pods() -> None:
    import runpod as rp
    rp.api_key = get_api_key()
    pods = rp.get_pods()
    if not pods:
        print("No running pods.")
        return
    for p in pods:
        status = p.get("desiredStatus", "?")
        name   = p.get("name", "?")
        ssh    = _ssh_info(p)
        print(f"  {p['id']}  {name:<35}  {status}  {ssh}")


def status_pod(pod_id: str) -> None:
    import runpod as rp
    rp.api_key = get_api_key()
    pod = rp.get_pod(pod_id)
    if not pod:
        sys.exit(f"Pod {pod_id} not found.")

    print(f"Pod ID  : {pod_id}")
    print(f"Name    : {pod.get('name', '?')}")
    print(f"Status  : {pod.get('desiredStatus', '?')}")
    print(f"GPU     : {pod.get('machine', {}).get('gpuDisplayName', '?')}")
    print()
    print(f"SSH     : {_ssh_info(pod)}")
    print()
    ports = (pod.get("runtime") or {}).get("ports") or []
    for port in ports:
        if port.get("privatePort") == 22 and port.get("isIpPublic"):
            print("~/.ssh/config entry:")
            print(f"  Host runpod-ice3")
            print(f"      HostName {port.get('ip', 'ssh.runpod.io')}")
            print(f"      User root")
            print(f"      Port {port['publicPort']}")
            print(f"      IdentityFile ~/.ssh/id_ed25519")


def wait_pod(pod_id: str, timeout: int = 300) -> None:
    import socket
    import time
    print(f"Waiting for pod {pod_id} SSH (timeout {timeout}s)...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            host, port = _get_pod_ssh_endpoint(pod_id)
            with socket.create_connection((host, port), timeout=5):
                print(f"Pod {pod_id} SSH ready at {host}:{port}")
                return
        except Exception:
            time.sleep(10)
    sys.exit(f"Timeout: pod {pod_id} SSH not reachable after {timeout}s.")


# ---------------------------------------------------------------------------
# bench fetch
# ---------------------------------------------------------------------------

def _maybe_fetch_bench(task: str, host: str, port: int, priv_key: str, ssh_e: str) -> None:
    if task not in BENCH_TASKS:
        return
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    local_dir = "benchmarks/"
    os.makedirs(local_dir, exist_ok=True)
    local_path = f"{local_dir}bench_{ts}.json"
    print(f"\nFetching bench results → {local_path}")
    result = subprocess.run([
        "rsync", "-avz", "-e", ssh_e,
        f"root@{host}:{_log_file('bench').replace('.log', '.json')}",
        local_path,
    ])
    if result.returncode == 0:
        print(f"  Saved: {local_path}")
    else:
        print("  WARNING: could not fetch bench.json (pod may still be running)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch / manage dwarf-p-ice3 GPU pods on RunPod",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Examples:",
            "  # Launch a smoke-gpu pod",
            "  python scripts/runpod_launch.py --task smoke-gpu",
            "",
            "  # Sync Python changes and re-run tests without a rebuild",
            "  python scripts/runpod_launch.py --sync <POD_ID> --task test-phyex",
            "",
            "  # Tail logs live",
            "  python scripts/runpod_launch.py --logs <POD_ID> --task test-physics",
        ]),
    )

    parser.add_argument("--task",   default="smoke-cpu", choices=VALID_TASKS,
                        help="Task to run on the pod (default: smoke-cpu)")
    parser.add_argument("--ssh-key", default="~/.ssh/id_ed25519.pub", metavar="PATH",
                        help="Path to SSH public key")
    parser.add_argument("--gpu",    default="A100", metavar="NAME",
                        help="GPU family to search (default: A100)")
    parser.add_argument("--image",  default=None, metavar="IMAGE",
                        help=f"Override container image (default: {IMAGE})")

    # Mutually-exclusive actions
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--stop",    metavar="POD_ID", help="Terminate a pod")
    g.add_argument("--status",  metavar="POD_ID", help="Show status and SSH info")
    g.add_argument("--list",    action="store_true", help="List running pods")
    g.add_argument("--wait",    metavar="POD_ID", help="Block until SSH is reachable")
    g.add_argument("--sync",    metavar="POD_ID",
                   help="Rsync local src/+tests/ to a running pod (add --task to also run)")
    g.add_argument("--logs",    metavar="POD_ID",
                   help="Tail the log for --task on a running pod")
    g.add_argument("--register-ghcr", action="store_true",
                   help="Register GHCR credentials with RunPod (one-time setup)")

    args = parser.parse_args()

    # Allow image override via --image flag
    if args.image:
        os.environ["ICE3_IMAGE"] = args.image

    try:
        import runpod  # noqa: F401
    except ImportError:
        sys.exit("ERROR: runpod not installed.\n  pip install runpod")

    if args.stop:
        stop_pod(args.stop)
    elif args.status:
        status_pod(args.status)
    elif args.list:
        list_pods()
    elif args.wait:
        wait_pod(args.wait)
    elif args.sync:
        task = args.task if "--task" in sys.argv else None
        sync_pod(pod_id=args.sync, ssh_key_path=args.ssh_key, task=task)
    elif args.logs:
        logs_pod(pod_id=args.logs, ssh_key_path=args.ssh_key, task=args.task)
    elif args.register_ghcr:
        register_ghcr()
    else:
        create_pod(task=args.task, ssh_key_path=args.ssh_key, gpu_name=args.gpu)


if __name__ == "__main__":
    main()
