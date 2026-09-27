"""Dashboard static files, sent with Cache-Control: no-cache so a normal reload always
checks for a newer app.js (browsers kept a stale one otherwise). An unchanged file still
comes back as a cheap 304, thanks to its ETag and Last-Modified headers."""

from __future__ import annotations

from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response
