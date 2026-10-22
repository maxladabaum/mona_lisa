from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import tile_checkpoint
from test_tile_checkpoint import run_fixture


def test_low_system_disk_stops_with_recovery_instructions(monkeypatch):
    monkeypatch.setattr(tile_checkpoint.sys, 'platform', 'darwin')
    monkeypatch.setattr(tile_checkpoint.shutil, 'disk_usage', lambda path: SimpleNamespace(free=187_000_000))
    with pytest.raises(tile_checkpoint.LowSystemDiskSpaceError, match='Completed tile checkpoints are retained'):
        tile_checkpoint.check_system_disk_headroom()


def test_sufficient_disk_space_allows_work(monkeypatch):
    monkeypatch.setattr(tile_checkpoint.sys, 'platform', 'darwin')
    monkeypatch.setattr(tile_checkpoint.shutil, 'disk_usage', lambda path: SimpleNamespace(free=4 * 1024 ** 3))
    tile_checkpoint.check_system_disk_headroom()


def test_resume_stops_before_processing_or_modifying_checkpoints(tmp_path, monkeypatch):
    import paint_analysis_gui as gui
    live, run, _ = run_fixture()
    saved = tmp_path / 'tile-00000000.paintanalysis'
    saved.write_bytes(b'existing checkpoint')
    monkeypatch.setattr(gui, 'check_system_disk_headroom', Mock(
        side_effect=tile_checkpoint.LowSystemDiskSpaceError('Free startup-disk space')))
    live._tiled_origami_worker = Mock()
    with pytest.raises(tile_checkpoint.LowSystemDiskSpaceError):
        live._checkpointed_tiled_worker(tmp_path, run)
    live._tiled_origami_worker.assert_not_called()
    assert saved.read_bytes() == b'existing checkpoint'
