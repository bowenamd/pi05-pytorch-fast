# Ubuntu + ROCm + PyTorch images checked on 2026-10-28

AMD publishes ready-to-pull **Ubuntu container images** with ROCm and PyTorch.
These are OCI/Docker images, not Ubuntu installation ISOs or VM disk images.
No official, verified preinstalled Ubuntu ISO/VM image with the exact
ROCm 10.1 + PyTorch 2.15 cohort was identified in this search.

| Image | Confirmed properties | Fit for this recipe |
|---|---|---|
| `rocm/pytorch:rocm10.1.0_ubuntu24.04_py3.12_pytorch_release_2.14.0` | AMD official; Ubuntu 24.04, Python 3.12, ROCm 10.1, PyTorch 2.14 | Ready-made ROCm 10.1 Ubuntu base, but torch differs from the pinned 2.15 trainer |
| `rocm/pytorch-nightly:20261019-90a0133` | AMD official; registry config labels say Ubuntu 24.04, ROCm **10.0**, PyTorch nightly commit `90a01338d9317d4b0b87caf478b933873deedee7` | A ready-made ROCm 10.x nightly; exact PyTorch version was not runtime-verified, and it is not the 10.1 cohort |
| `rocm/pytorch-nightly:20260902-3c73a85` | Historical tag exists in the official registry | Existence alone does not confirm its internal ROCm/torch versions |

Registry digests checked:

```text
rocm/pytorch:rocm10.1.0_ubuntu24.04_py3.12_pytorch_release_2.14.0
sha256:099f59a3bee9c436f5cc57db6bc6076fdeda579c57e9e22ae40f856d5905b3e4

rocm/pytorch-nightly:20261019-90a0133
sha256:4fbfa18df1c02b00e839c131b611f69bef9c99cb1959162d67c3251552186f9c
```

The nightly image's inspected config labels explicitly contain:

```text
org.opencontainers.image.version=24.04
rocm.version=10.0
rocm.package_index=https://stable.repo.amd.com/rocm/whl-next/
torch.package_index=https://download.pytorch.org/whl/nightly/rocm10.0/
pytorch.branch=nightly
```

## Pull and check an official release container

On a host with Docker and a working AMD GPU driver:

```bash
docker pull rocm/pytorch@sha256:099f59a3bee9c436f5cc57db6bc6076fdeda579c57e9e22ae40f856d5905b3e4

docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add "$(stat -c '%g' /dev/kfd)" --shm-size=8g \
  rocm/pytorch@sha256:099f59a3bee9c436f5cc57db6bc6076fdeda579c57e9e22ae40f856d5905b3e4 \
  python -c 'import torch; print(torch.__version__, torch.version.hip); print(torch.cuda.is_available())'
```

Containers share the host's kernel driver; they do not install it. Image size is
also material: the queried nightly tag is about 16.76 GB compressed.

For customer delivery requiring the **exact 2.15 stack**, build an organization-owned
image from Ubuntu 24.04 using the from-zero recipe and pinned `rocm_stack.json`.
Run the GPU/training smoke on the target hardware, then publish and reference the
organization's immutable image digest. A moving `latest` tag or a community image
on an individual's account is not a substitute for that tested release.

This check read registry tags, the nightly image manifest/config, and official
documentation. It did not pull or execute the full Docker images. The conda-based
setup and training validation are recorded separately in `VERIFICATION.md`.

## Sources

- [AMD PyTorch installation guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)
- [Official release image tags](https://hub.docker.com/r/rocm/pytorch/tags)
- [Official nightly image tags](https://hub.docker.com/r/rocm/pytorch-nightly/tags)
- [AMD multi-architecture PyTorch wheel index](https://rocm.nightlies.amd.com/whl-multi-arch/torch/)
- [Official ROCm 10.1 host/SDK installation guide](https://rocm.docs.amd.com/en/docs-10.1.0/install/rocm.html)
