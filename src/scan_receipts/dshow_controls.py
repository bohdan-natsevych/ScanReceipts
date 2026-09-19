from __future__ import annotations

import contextlib
from ctypes import POINTER, byref, c_int, c_long, c_ulong, c_ulonglong, c_void_p, cast
from dataclasses import dataclass, replace
from typing import Any, ClassVar

from .models import CameraCapability

try:
    import comtypes
    from comtypes import COMMETHOD, GUID, HRESULT, IUnknown
    from comtypes.automation import VARIANT
    from comtypes.persist import IPropertyBag
except (ImportError, OSError):  # pragma: no cover - non-Windows or missing package
    comtypes = None  # type: ignore[assignment]

CONTROL_NAMES = ("brightness", "contrast", "white_balance", "zoom", "exposure", "focus")
_VIDEO_PROC_AMP = {"brightness": 0, "contrast": 1, "white_balance": 7}
_CAMERA_CONTROL = {"zoom": 3, "exposure": 4, "focus": 6}
_FLAG_AUTO = 0x1


@dataclass(frozen=True, slots=True)
class ControlRange:
    minimum: int
    maximum: int
    step: int
    default: int
    supports_auto: bool
    current: int
    auto_active: bool


if comtypes is not None:
    CLSID_SystemDeviceEnum = GUID("{62BE5D10-60EB-11d0-BD3B-00A0C911CE86}")
    CLSID_VideoInputDeviceCategory = GUID("{860BB310-5D01-11d0-BD3B-00A0C911CE86}")
    IID_IBaseFilter = GUID("{56A86895-0AD4-11CE-B03A-0020AF0BA770}")

    class IMoniker(IUnknown):
        _iid_ = GUID("{0000000F-0000-0000-C000-000000000046}")
        # CURSOR: vtable order must include the IPersistStream slots that
        # precede BindToObject; unused ones are declared as stubs.
        _methods_: ClassVar[list[Any]] = [
            COMMETHOD([], HRESULT, "GetClassID", (["out"], POINTER(GUID), "pClassID")),
            COMMETHOD([], HRESULT, "IsDirty"),
            COMMETHOD([], HRESULT, "Load", (["in"], c_void_p, "pStm")),
            COMMETHOD(
                [],
                HRESULT,
                "Save",
                (["in"], c_void_p, "pStm"),
                (["in"], c_int, "fClearDirty"),
            ),
            COMMETHOD(
                [], HRESULT, "GetSizeMax", (["out"], POINTER(c_ulonglong), "pcbSize")
            ),
            COMMETHOD(
                [],
                HRESULT,
                "BindToObject",
                (["in"], c_void_p, "pbc"),
                (["in"], c_void_p, "pmkToLeft"),
                (["in"], POINTER(GUID), "riidResult"),
                (["out"], POINTER(c_void_p), "ppvResult"),
            ),
            COMMETHOD(
                [],
                HRESULT,
                "BindToStorage",
                (["in"], c_void_p, "pbc"),
                (["in"], c_void_p, "pmkToLeft"),
                (["in"], POINTER(GUID), "riid"),
                (["out"], POINTER(c_void_p), "ppvObj"),
            ),
        ]

    class IEnumMoniker(IUnknown):
        _iid_ = GUID("{00000102-0000-0000-C000-000000000046}")
        _methods_: ClassVar[list[Any]] = [
            COMMETHOD(
                [],
                HRESULT,
                "Next",
                (["in"], c_ulong, "celt"),
                (["out"], POINTER(POINTER(IMoniker)), "rgelt"),
                (["out"], POINTER(c_ulong), "pceltFetched"),
            ),
            COMMETHOD([], HRESULT, "Skip", (["in"], c_ulong, "celt")),
            COMMETHOD([], HRESULT, "Reset"),
            COMMETHOD(
                [], HRESULT, "Clone", (["out"], POINTER(POINTER(IUnknown)), "ppenum")
            ),
        ]

    class ICreateDevEnum(IUnknown):
        _iid_ = GUID("{29840822-5B84-11D0-BD3B-00A0C911CE86}")
        _methods_: ClassVar[list[Any]] = [
            COMMETHOD(
                [],
                HRESULT,
                "CreateClassEnumerator",
                (["in"], POINTER(GUID), "clsidDeviceClass"),
                (["out"], POINTER(POINTER(IEnumMoniker)), "ppEnumMoniker"),
                (["in"], c_ulong, "dwFlags"),
            ),
        ]

    class IAMVideoProcAmp(IUnknown):
        _iid_ = GUID("{C6E13360-30AC-11d0-A18C-00A0C9118956}")
        _methods_: ClassVar[list[Any]] = [
            COMMETHOD(
                [],
                HRESULT,
                "GetRange",
                (["in"], c_long, "property_id"),
                (["out"], POINTER(c_long), "minimum"),
                (["out"], POINTER(c_long), "maximum"),
                (["out"], POINTER(c_long), "stepping"),
                (["out"], POINTER(c_long), "default"),
                (["out"], POINTER(c_long), "cap_flags"),
            ),
            COMMETHOD(
                [],
                HRESULT,
                "Set",
                (["in"], c_long, "property_id"),
                (["in"], c_long, "value"),
                (["in"], c_long, "flags"),
            ),
            COMMETHOD(
                [],
                HRESULT,
                "Get",
                (["in"], c_long, "property_id"),
                (["out"], POINTER(c_long), "value"),
                (["out"], POINTER(c_long), "flags"),
            ),
        ]

    class IAMCameraControl(IUnknown):
        _iid_ = GUID("{C6E13370-30AC-11d0-A18C-00A0C9118956}")
        # CURSOR: same vtable shape as IAMVideoProcAmp, but copied so the two
        # interface classes never share one mutable list.
        _methods_: ClassVar[list[Any]] = list(IAMVideoProcAmp._methods_)


def _read_ranges(interface: Any, mapping: dict[str, int]) -> dict[str, ControlRange]:
    found: dict[str, ControlRange] = {}
    for name, property_id in mapping.items():
        try:
            minimum, maximum, step, default, caps = interface.GetRange(property_id)
            current, flags = interface.Get(property_id)
        except Exception:
            continue
        found[name] = ControlRange(
            minimum=int(minimum),
            maximum=int(maximum),
            step=int(step),
            default=int(default),
            supports_auto=bool(int(caps) & _FLAG_AUTO),
            current=int(current),
            auto_active=bool(int(flags) & _FLAG_AUTO),
        )
    return found


def control_ranges(device_name: str) -> dict[str, ControlRange]:
    """Real min/max/step/auto capability per control, or {} when unavailable."""
    if comtypes is None:
        return {}
    initialized = False
    try:
        comtypes.CoInitialize()
        initialized = True
    except Exception:
        # The thread already lives in an incompatible apartment; the calls
        # below still work there, and it is not ours to uninitialize.
        pass
    try:
        return _device_control_ranges(device_name)
    finally:
        # CURSOR: every successful init needs its own uninit, and only after
        # the interface pointers of the inner call have been released.
        if initialized:
            with contextlib.suppress(Exception):
                comtypes.CoUninitialize()


def _device_control_ranges(device_name: str) -> dict[str, ControlRange]:
    try:
        enum = comtypes.CoCreateInstance(
            CLSID_SystemDeviceEnum,
            interface=ICreateDevEnum,
            clsctx=comtypes.CLSCTX_INPROC_SERVER,
        )
        class_enum = enum.CreateClassEnumerator(
            byref(CLSID_VideoInputDeviceCategory), 0
        )
        if not class_enum:
            return {}
        while True:
            moniker, fetched = class_enum.Next(1)
            if not fetched or not moniker:
                return {}
            if _friendly_name(moniker) == device_name:
                break
        pointer = moniker.BindToObject(None, None, byref(IID_IBaseFilter))
        unknown = cast(pointer, POINTER(IUnknown))
        ranges: dict[str, ControlRange] = {}
        with contextlib.suppress(Exception):
            ranges.update(
                _read_ranges(unknown.QueryInterface(IAMVideoProcAmp), _VIDEO_PROC_AMP)
            )
        with contextlib.suppress(Exception):
            ranges.update(
                _read_ranges(unknown.QueryInterface(IAMCameraControl), _CAMERA_CONTROL)
            )
        return ranges
    except Exception:
        return {}


def _friendly_name(moniker: Any) -> str:
    try:
        pointer = moniker.BindToStorage(None, None, byref(IPropertyBag._iid_))
        bag = cast(pointer, POINTER(IPropertyBag))
        variant = VARIANT()
        bag.Read("FriendlyName", byref(variant), None)
        return str(variant.value)
    except Exception:
        return ""


def merge_capabilities(
    base: list[CameraCapability], ranges: dict[str, ControlRange]
) -> list[CameraCapability]:
    """Overlay real DirectShow ranges onto the OpenCV capability heuristic.

    CURSOR: pure - the caller's capabilities are copied, never mutated, so a
    cached capability list stays valid after a merge.
    """
    auto_sources = {
        "autofocus": "focus",
        "auto_exposure": "exposure",
        "auto_white_balance": "white_balance",
    }
    merged: list[CameraCapability] = []
    for capability in base:
        control = ranges.get(capability.name)
        if control is not None:
            entry = replace(
                capability,
                supported=True,
                minimum=float(control.minimum),
                maximum=float(control.maximum),
                value=(
                    float(control.current)
                    if capability.value is None
                    else capability.value
                ),
            )
        elif ranges and capability.name in CONTROL_NAMES:
            entry = replace(capability, supported=False)
        elif ranges and capability.name in auto_sources:
            source = ranges.get(auto_sources[capability.name])
            entry = replace(capability, supported=bool(source and source.supports_auto))
        else:
            entry = replace(capability)
        merged.append(entry)
    return merged
