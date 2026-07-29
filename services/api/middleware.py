from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

log = structlog.get_logger()

# These paths skip auth — health check and docs should always be accessible
PUBLIC_PATHS = {
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Checks every request for a valid API key in the X-API-Key header.

    WHY MIDDLEWARE AND NOT A DEPENDENCY?
    A FastAPI Depends() would need to be added to every single endpoint.
    Middleware runs automatically on every request — one place, covers all routes.
    We just whitelist the public paths that don't need auth.
    """

    def __init__(self, app, api_key: str):
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next):
        # Skip auth for public paths
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        # Check for API key header
        provided_key = request.headers.get("X-API-Key")

        if not provided_key:
            log.warning("auth.missing_key", path=request.url.path)
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing X-API-Key header"}
            )

        if provided_key != self.api_key:
            log.warning("auth.invalid_key", path=request.url.path)
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid API key"}
            )

        # Key is valid — proceed with the request
        return await call_next(request)