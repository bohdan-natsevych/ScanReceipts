"""Single source of the running application version.

CLAUDE CODE: the release workflow rewrites APP_VERSION before freezing the app;
the committed value is only what a source checkout reports.
"""

from __future__ import annotations

APP_VERSION = "0.1.0"
