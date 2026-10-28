"""Validate the installed training stack and, by default, actual GPU computation."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--skip-gpu', action='store_true')
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--arch', default='gfx1151')
    args = p.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Python 3.12 is required')
    lock = json.loads((Path(__file__).parent / 'rocm_stack.json').read_text())
    for name, version in lock['packages'].items():
        if metadata.version(name) != version:
            raise RuntimeError(f'{name}: expected {version}, got {metadata.version(name)}')
    import torch
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: F401
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
    from lerobot.policies.pi05 import processor_pi05  # noqa: F401
    import av
    import transformers
    if metadata.version('lerobot') != '0.6.1' or transformers.__version__ != '5.5.4':
        raise RuntimeError('Unexpected LeRobot/Transformers version')
    if not torch.version.hip:
        raise RuntimeError('Installed torch is not a ROCm build')
    result = {'python': sys.executable, 'torch': torch.__version__, 'hip': torch.version.hip,
              'rocm_sdk': metadata.version('rocm'), 'transformers': transformers.__version__,
              'av': av.__version__, 'gpu_tested': False}
    if not args.skip_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError('AMD GPU unavailable. Check amdgpu driver, /dev/kfd, /dev/dri and render/video groups')
        properties = torch.cuda.get_device_properties(0)
        detected = properties.gcnArchName.split(':')[0]
        if detected != args.arch:
            raise RuntimeError(f'Installed architecture {args.arch} does not match GPU {detected}')
        generator = torch.Generator().manual_seed(42)
        a = torch.randn(128, 128, generator=generator).to(torch.bfloat16)
        b = torch.randn(128, 128, generator=generator).to(torch.bfloat16)
        observed = (a.cuda() @ b.cuda()).float().cpu()
        expected = a.float() @ b.float()
        torch.testing.assert_close(observed, expected, atol=0.2, rtol=0.02)
        result.update(gpu_tested=True, gpu=properties.name, arch=detected, bf16_matmul=True)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
