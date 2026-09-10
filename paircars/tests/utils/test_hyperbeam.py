import numpy as np
import pytest
import pickle
from paircars.utils.hyperbeam import FEEBeam


@pytest.mark.parametrize(
    "method_name, check_container, container_exists",
    [
        ("calc_jones_array", False, True),
        ("calc_jones_array", True, True),
        ("calc_jones_array", True, False),
        ("calc_jones", False, True),
        ("calc_jones", True, True),
    ],
)
def test_feebeam_all(
    monkeypatch,
    method_name,
    check_container,
    container_exists,
):
    # Mock udocker init
    monkeypatch.setattr("paircars.utils.hyperbeam.init_udocker", lambda: None)
    # Mock container check
    monkeypatch.setattr(
        "paircars.utils.hyperbeam.check_udocker_container",
        lambda name: container_exists,
    )
    # Mock container initialization
    monkeypatch.setattr(
        "paircars.utils.hyperbeam.initialize_hyperbeam_container",
        lambda name, verbose=True: name,
    )
    # Capture subprocess call
    captured = {}

    class DummyProc:
        def __init__(self, stdout):
            self.stdout = stdout

    def fake_run(cmd, env=None, input=None, capture_output=None):
        captured["cmd"] = cmd
        captured["env"] = env
        captured["input"] = input
        # Validate input is pickled
        data = pickle.loads(input)
        assert "az_rad" in data
        assert "pbfile" in data
        # Return fake result
        fake_output = np.ones((2, 2), dtype=np.float32)
        return DummyProc(stdout=pickle.dumps(fake_output))

    monkeypatch.setattr("paircars.utils.hyperbeam.subprocess.run", fake_run)
    pbfile = "/tmp/fake_pb.h5"
    beam = FEEBeam(pbfile, check_container=check_container)
    args = dict(
        az_rad=np.array([0.1, 0.2]),
        za_rad=np.array([0.3, 0.4]),
        freq=1e8,
        delay=[0] * 16,
        amps=[1.0] * 16,
        norm=True,
        lat=-26.7,
        iau_order=True,
    )
    method = getattr(beam, method_name)
    result = method(**args)
    # ---------------- ASSERT ---------------- #
    # Output check
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    # Command sanity
    assert "udocker" in captured["cmd"][0]
    if method_name == "calc_jones_array":
        assert any("hyperbeam_array.py" in x for x in captured["cmd"])
    else:
        assert any("hyperbeam_single.py" in x for x in captured["cmd"])
    # Env propagated
    assert captured["env"] is not None
    # Input was passed
    assert captured["input"] is not None
