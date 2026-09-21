import numpy as np

from scan_receipts.camera import OpenCVSource, _parse_dshow_device_names
from scan_receipts.config import default_settings


def test_directshow_names_are_parsed_without_alternative_ids() -> None:
    output = """
[dshow @ 0001] "Integrated Camera" (video)
[dshow @ 0001]   Alternative name "@device_pnp_..."
[dshow @ 0001] "My Phone (Windows Virtual Camera)" (video)
[dshow @ 0001] "Microphone Array" (audio)
"""

    assert _parse_dshow_device_names(output) == [
        "Integrated Camera",
        "My Phone (Windows Virtual Camera)",
    ]


MJPG = 1196444237  # cv2.VideoWriter_fourcc(*"MJPG")


class FakeCapture:
    """Stands in for cv2.VideoCapture, recording the order properties are set."""

    def __init__(self, *, mjpg_delivers: bool = True) -> None:
        self.mjpg_delivers = mjpg_delivers
        self.calls: list[tuple[int, float]] = []
        self.props: dict[int, float] = {
            FakeCv2.CAP_PROP_FRAME_WIDTH: 1920.0,
            FakeCv2.CAP_PROP_FRAME_HEIGHT: 1080.0,
            FakeCv2.CAP_PROP_FPS: 30.0,
            FakeCv2.CAP_PROP_FOURCC: 844715353.0,  # YUY2
        }
        self.released = False
        self.reads = 0

    def isOpened(self) -> bool:  # noqa: N802
        return True

    def set(self, prop: int, value: float) -> bool:
        self.calls.append((prop, value))
        self.props[prop] = value
        return True

    def get(self, prop: int) -> float:
        return self.props.get(prop, 0.0)

    def read(self):
        self.reads += 1
        if self.props.get(FakeCv2.CAP_PROP_FOURCC) == MJPG and not self.mjpg_delivers:
            return False, None
        return True, np.zeros((4, 4, 3), np.uint8)

    def release(self) -> None:
        self.released = True


class FakeCv2:
    error = type("error", (Exception,), {})
    CAP_ANY = 0
    CAP_MSMF = 1400
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_FOURCC = 6
    CAP_PROP_AUTOFOCUS = 39

    def __init__(self, *, mjpg_delivers: bool = True) -> None:
        self.mjpg_delivers = mjpg_delivers
        self.opened: list[FakeCapture] = []

    def VideoCapture(self, index, backend=None):  # noqa: N802
        capture = FakeCapture(mjpg_delivers=self.mjpg_delivers)
        self.opened.append(capture)
        return capture

    @staticmethod
    def VideoWriter_fourcc(*code):  # noqa: N802
        return MJPG


def camera_source(monkeypatch, *, mjpg_delivers: bool = True):
    fake = FakeCv2(mjpg_delivers=mjpg_delivers)
    monkeypatch.setattr("scan_receipts.camera.cv2", fake)
    settings = default_settings()
    return OpenCVSource(0, settings, name="Test camera"), fake


def test_a_usb_camera_is_asked_for_mjpg_before_its_resolution(monkeypatch) -> None:
    source, fake = camera_source(monkeypatch)

    source.open()

    capture = fake.opened[0]
    order = [prop for prop, _value in capture.calls]
    assert FakeCv2.CAP_PROP_FOURCC in order, "MJPG must be requested"
    assert capture.props[FakeCv2.CAP_PROP_FOURCC] == MJPG
    assert order.index(FakeCv2.CAP_PROP_FOURCC) < order.index(
        FakeCv2.CAP_PROP_FRAME_WIDTH
    ), "the format must be negotiated before the resolution"


def test_opening_a_camera_never_reads_a_frame(monkeypatch) -> None:
    """Probing with read() during open cost an access violation inside OpenCV."""
    source, fake = camera_source(monkeypatch)

    source.open()

    assert fake.opened[0].reads == 0, "open() must not pull frames from the device"


def test_a_camera_that_refuses_mjpg_still_opens(monkeypatch) -> None:
    source, fake = camera_source(monkeypatch, mjpg_delivers=False)

    source.open()

    assert len(fake.opened) == 1, "one open, no retry storm on the same device"
    assert source.read() is None


def test_a_driver_that_rejects_the_format_does_not_lose_the_camera(monkeypatch) -> None:
    """cv2.set can raise on its own; her log shows exactly this from _attempt."""
    source, fake = camera_source(monkeypatch)
    capture = None

    original = FakeCapture.set

    def explode_on_fourcc(self, prop, value):
        if prop == FakeCv2.CAP_PROP_FOURCC:
            raise FakeCv2.error("Unknown C++ exception from OpenCV code")
        return original(self, prop, value)

    monkeypatch.setattr(FakeCapture, "set", explode_on_fourcc)

    source.open()

    capture = fake.opened[0]
    assert not capture.released, "a refused format must not cost us the camera"
    assert capture.props[FakeCv2.CAP_PROP_FRAME_WIDTH] == 1920
