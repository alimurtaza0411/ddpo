#!/usr/bin/env python3
"""
provision_gpu.py — provision a single GPU VM on GCP for DDPO training.

Tries each zone in order until one accepts the VM creation request. Prints
the chosen zone and instance name on success so the caller can ssh / scp.

Defaults (overridable via flags):
  project       dist-sys-472800
  gpu_type      nvidia-tesla-v100 (project has no A100 quota; V100 is the
                fastest available alternative without quota requests)
  machine_type  n1-standard-8        (8 vCPU, 30GB RAM, 1× V100)
  disk_size     150 GB pd-balanced
  boot_image    ubuntu-2204-lts (plain Ubuntu, we install CUDA 12.1 + PyTorch 2.4 ourselves)
  spot          off (use --spot to enable)

Example:
    python scripts/provision_gpu.py --name ddpo-v100
    python scripts/provision_gpu.py --name ddpo-v100 --spot
    python scripts/provision_gpu.py --gpu-type nvidia-l4 --name ddpo-l4
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from google.cloud import compute_v1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("provision_gpu")


# Default zones — regions where Cloud NAT exists in this project
# (us-central1 and us-east1, set up earlier). VMs have no external IP per
# org policy, so NAT is required for pip/HF/wandb egress.
#
# Map of GPU type → preferred zone list (zones where that GPU is offered).
GPU_ZONES = {
    "nvidia-l4": [
        "us-east1-b", "us-east1-c", "us-east1-d",
        "us-central1-a", "us-central1-b", "us-central1-c",
    ],
    "nvidia-tesla-v100": [
        "us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f",
        "us-east1-c",
    ],
    "nvidia-tesla-t4": [
        "us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f",
        "us-east1-c", "us-east1-d",
    ],
    "nvidia-tesla-a100": [
        "us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f",
        "us-east1-b",
    ],
}

# Map of GPU type → required machine type family.
GPU_MACHINE = {
    "nvidia-l4":          "g2-standard-8",   # 1× L4, 8 vCPU, 32GB
    "nvidia-tesla-v100":  "n1-standard-8",   # 1× V100, 8 vCPU, 30GB
    "nvidia-tesla-t4":    "n1-standard-8",   # 1× T4,   8 vCPU, 30GB
    "nvidia-tesla-a100":  "a2-highgpu-1g",   # 1× A100, 12 vCPU, 85GB
}

# Backward-compat name kept for older callers.
DEFAULT_L4_ZONES = GPU_ZONES["nvidia-l4"]


@dataclass
class Config:
    project: str = "dist-sys-472800"
    gpu_type: str = "nvidia-tesla-v100"
    machine_type: str = "n1-standard-8"
    disk_size_gb: int = 150
    disk_type: str = "pd-balanced"
    # Plain Ubuntu 22.04 LTS, no preinstalled drivers/PyTorch.
    # The deep-learning image (cu129) had a fundamental cuBLAS Lt incompatibility
    # with our SD 1.5 workload — see scripts/install_cuda_pytorch.sh for
    # the cu121 stack we install manually instead.
    boot_image: str = (
        "projects/ubuntu-os-cloud/global/images/family/ubuntu-2204-lts"
    )
    gpu_count: int = 1
    spot: bool = False
    instance_name: str = "ddpo-v100"


def build_instance(cfg: Config, zone: str) -> compute_v1.Instance:
    region = "-".join(zone.split("-")[:-1])

    disk = compute_v1.AttachedDisk(
        auto_delete=True,
        boot=True,
        initialize_params=compute_v1.AttachedDiskInitializeParams(
            source_image=cfg.boot_image,
            disk_size_gb=cfg.disk_size_gb,
            disk_type=f"projects/{cfg.project}/zones/{zone}/diskTypes/{cfg.disk_type}",
        ),
    )

    # No external IP (org policy in this project forbids it). SSH happens via
    # IAP tunneling, which gcloud's `--tunnel-through-iap` flag handles.
    network = compute_v1.NetworkInterface(
        stack_type="IPV4_ONLY",
        subnetwork=f"projects/{cfg.project}/regions/{region}/subnetworks/default",
    )

    accelerator = compute_v1.AcceleratorConfig(
        accelerator_count=cfg.gpu_count,
        accelerator_type=(
            f"projects/{cfg.project}/zones/{zone}/acceleratorTypes/{cfg.gpu_type}"
        ),
    )

    scheduling = compute_v1.Scheduling(
        provisioning_model="SPOT" if cfg.spot else "STANDARD",
        automatic_restart=False if cfg.spot else True,
        on_host_maintenance="TERMINATE",
        preemptible=cfg.spot,
    )

    # No deeplearning-image driver-install hook on plain Ubuntu —
    # scripts/install_cuda_pytorch.sh does the install instead.
    return compute_v1.Instance(
        name=cfg.instance_name,
        machine_type=(
            f"projects/{cfg.project}/zones/{zone}/machineTypes/{cfg.machine_type}"
        ),
        guest_accelerators=[accelerator],
        scheduling=scheduling,
        disks=[disk],
        network_interfaces=[network],
    )


def try_zone(cfg: Config, zone: str, timeout: int = 600) -> Tuple[bool, Optional[str]]:
    instance = build_instance(cfg, zone)
    client = compute_v1.InstancesClient()
    try:
        op = client.insert(
            request=compute_v1.InsertInstanceRequest(
                project=cfg.project,
                zone=zone,
                instance_resource=instance,
            )
        )
        op.result(timeout=timeout)
        return True, None
    except Exception as exc:
        return False, str(exc).splitlines()[0]


def provision(cfg: Config, zones: List[str]) -> Optional[str]:
    logger.info(
        "Provisioning %s in project=%s (machine=%s, spot=%s) — trying %d zones",
        cfg.gpu_type, cfg.project, cfg.machine_type, cfg.spot, len(zones),
    )
    for zone in zones:
        logger.info("Trying zone %s ...", zone)
        t0 = time.time()
        ok, err = try_zone(cfg, zone)
        if ok:
            logger.info(
                "✅ Created %s in %s (%.1fs)", cfg.instance_name, zone, time.time() - t0
            )
            return zone
        logger.info("  ✗ %s rejected: %s", zone, err)
    logger.error("❌ All %d zones rejected the request", len(zones))
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default="dist-sys-472800")
    p.add_argument("--name", default="ddpo-v100", help="VM instance name")
    p.add_argument(
        "--gpu-type",
        default="nvidia-tesla-v100",
        choices=list(GPU_ZONES.keys()),
        help="GPU type. Default V100 (no A100 quota in this project).",
    )
    p.add_argument(
        "--machine-type",
        default=None,
        help="Override machine type. If omitted, derived from --gpu-type via GPU_MACHINE.",
    )
    p.add_argument("--disk-size", type=int, default=150)
    p.add_argument("--spot", action="store_true", help="Use spot/preemptible (cheaper, can be interrupted)")
    p.add_argument(
        "--zones",
        default=None,
        help="Comma-separated zone list to try in order. If omitted, uses GPU_ZONES[gpu_type].",
    )
    args = p.parse_args()

    zones = (
        [z.strip() for z in args.zones.split(",") if z.strip()]
        if args.zones
        else GPU_ZONES.get(args.gpu_type, DEFAULT_L4_ZONES)
    )

    machine_type = args.machine_type or GPU_MACHINE.get(args.gpu_type, "n1-standard-8")

    cfg = Config(
        project=args.project,
        gpu_type=args.gpu_type,
        machine_type=machine_type,
        disk_size_gb=args.disk_size,
        spot=args.spot,
        instance_name=args.name,
    )

    chosen_zone = provision(cfg, zones)
    if chosen_zone is None:
        return 1

    print()
    print("=" * 70)
    print(f"  Instance:  {cfg.instance_name}")
    print(f"  Zone:      {chosen_zone}")
    print(f"  Project:   {cfg.project}")
    print()
    print("Next steps:")
    print(f"  ./scripts/sync_to_vm.sh {cfg.instance_name} {chosen_zone}")
    print(f"  gcloud compute ssh {cfg.instance_name} --zone={chosen_zone} --project={cfg.project}")
    print(f"  # then on the VM:  bash ~/RL-project/scripts/bootstrap_vm.sh")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
