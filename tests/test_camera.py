from scan_receipts.camera import _parse_dshow_device_names


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
