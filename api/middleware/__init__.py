"""
API middleware for Movie Recommendation System.
"""

import time
import logging
from fastapi import Request, Response
from typing import Callable
import json

logger = logging.getLogger(__name__)

class RequestLoggingMiddleware:
    """Middleware for logging HTTP requests."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        
        start_time = time.time()
        
        # Create custom send wrapper to capture response
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                # Store status code
                scope["status_code"] = message["status"]
            await send(message)
        
        # Process request
        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            # Log unhandled exceptions
            duration = time.time() - start_time
            logger.error(
                f"Request error: {scope['method']} {scope['path']} "
                f"- Duration: {duration:.3f}s - Error: {str(e)}"
            )
            raise
        
        # Log successful request
        duration = time.time() - start_time
        status_code = scope.get("status_code", 500)
        
        logger.info(
            f"{scope['method']} {scope['path']} "
            f"- Status: {status_code} - Duration: {duration:.3f}s"
        )

class AuthenticationMiddleware:
    """Middleware for API authentication."""
    
    def __init__(self, app, api_keys: set = None):
        self.app = app
        self.api_keys = api_keys or set()
        self.public_endpoints = {
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json"
        }
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        
        # Check if endpoint is public
        path = scope["path"]
        if path in self.public_endpoints or path.startswith("/static"):
            return await self.app(scope, receive, send)
        
        # Extract API key from headers
        headers = dict(scope["headers"])
        api_key = None
        
        # Check for API key in headers
        for key, value in headers.items():
            if key.decode().lower() == "x-api-key":
                api_key = value.decode()
                break
        
        # Authenticate
        if not api_key or api_key not in self.api_keys:
            # Send 401 Unauthorized response
            response = {
                "status": "error",
                "message": "Invalid or missing API key"
            }
            
            await send({
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                ]
            })
            
            await send({
                "type": "http.response.body",
                "body": json.dumps(response).encode(),
            })
            return
        
        # Proceed with authenticated request
        return await self.app(scope, receive, send)

class RateLimitingMiddleware:
    """Middleware for rate limiting."""
    
    def __init__(self, app, rate_limit: int = 100, window_seconds: int = 60):
        self.app = app
        self.rate_limit = rate_limit
        self.window_seconds = window_seconds
        self.request_counts = {}
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        
        # Extract client IP
        client_ip = None
        for header, value in scope.get("headers", []):
            if header == b"x-forwarded-for":
                client_ip = value.decode().split(",")[0].strip()
                break
        
        if not client_ip:
            # Try to get from peer
            client_ip = scope.get("client", ["unknown"])[0]
        
        # Apply rate limiting
        current_time = time.time()
        window_key = f"{client_ip}:{int(current_time / self.window_seconds)}"
        
        # Get or initialize count
        if window_key not in self.request_counts:
            self.request_counts[window_key] = 0
        
        # Check rate limit
        if self.request_counts[window_key] >= self.rate_limit:
            # Send 429 Too Many Requests response
            response = {
                "status": "error",
                "message": "Rate limit exceeded",
                "retry_after": self.window_seconds
            }
            
            await send({
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"retry-after", str(self.window_seconds).encode())
                ]
            })
            
            await send({
                "type": "http.response.body",
                "body": json.dumps(response).encode(),
            })
            return
        
        # Increment count and proceed
        self.request_counts[window_key] += 1
        
        # Clean up old entries (optional, for memory management)
        self._cleanup_old_entries(current_time)
        
        return await self.app(scope, receive, send)
    
    def _cleanup_old_entries(self, current_time: float):
        """Clean up old rate limiting entries."""
        cutoff = current_time - (self.window_seconds * 2)
        cutoff_key = int(cutoff / self.window_seconds)
        
        # Remove entries older than cutoff
        keys_to_remove = []
        for key in self.request_counts.keys():
            window_id = int(key.split(":")[1])
            if window_id < cutoff_key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.request_counts[key]

class CORSMiddleware:
    """Middleware for CORS headers."""
    
    def __init__(self, app, allow_origins: list = None, 
                 allow_methods: list = None, allow_headers: list = None):
        self.app = app
        self.allow_origins = allow_origins or ["*"]
        self.allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allow_headers = allow_headers or ["*"]
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        
        # Handle preflight requests
        if scope["method"] == "OPTIONS":
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": self._get_cors_headers(scope)
            })
            await send({
                "type": "http.response.body",
                "body": b"",
            })
            return
        
        # For regular requests, add CORS headers
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                # Add CORS headers
                headers = dict(message.get("headers", []))
                cors_headers = self._get_cors_headers(scope)
                headers.update(cors_headers)
                
                # Convert headers back to list
                message["headers"] = [(k.encode(), v.encode()) 
                                     for k, v in headers.items()]
            
            await send(message)
        
        return await self.app(scope, receive, send_wrapper)
    
    def _get_cors_headers(self, scope) -> dict:
        """Get CORS headers based on request origin."""
        origin = None
        
        # Extract origin from headers
        for header, value in scope.get("headers", []):
            if header == b"origin":
                origin = value.decode()
                break
        
        headers = {}
        
        # Set Access-Control-Allow-Origin
        if origin and origin in self.allow_origins:
            headers["Access-Control-Allow-Origin"] = origin
        elif "*" in self.allow_origins:
            headers["Access-Control-Allow-Origin"] = "*"
        
        # Set other CORS headers
        headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
        headers["Access-Control-Allow-Headers"] = ", ".join(self.allow_headers)
        headers["Access-Control-Allow-Credentials"] = "true"
        headers["Access-Control-Max-Age"] = "86400"  # 24 hours
        
        return headers

class ErrorHandlingMiddleware:
    """Middleware for handling errors gracefully."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        
        try:
            return await self.app(scope, receive, send)
        except Exception as e:
            logger.error(f"Unhandled error: {str(e)}", exc_info=True)
            
            # Send error response
            response = {
                "status": "error",
                "message": "Internal server error",
                "error_code": "INTERNAL_ERROR"
            }
            
            await send({
                "type": "http.response.start",
                "status": 500,
                "headers": [
                    (b"content-type", b"application/json"),
                ]
            })
            
            await send({
                "type": "http.response.body",
                "body": json.dumps(response).encode(),
            })
