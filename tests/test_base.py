from src.warmup_scheduler_pytorch import WarmUpScheduler, __version__
from src.warmup_scheduler_pytorch.warmup_module import VERSION


def test_version():
    assert VERSION == __version__


def test_import():
    assert isinstance(WarmUpScheduler, object)
