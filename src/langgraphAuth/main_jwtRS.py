import os
import sys
import uuid
import asyncio
import datetime
import jwt
import uvicorn
from typing import Any, Optional
from contextlib import asynccontextmanager

from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from dotenv import load_dotenv, find_dotenv
from jwcrypto import jwk
from jwt import PyJWK

from langchain_google_genai import ChatGoogleGenerativeAI

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, InMemoryPushNotificationConfigStore, BasePushNotificationSender
from a2a.types import (
    AgentCapabilities, AgentCard, AgentSkill,
    SecurityScheme, HTTPAuthSecurityScheme
)

from weather_executor import WeatherExecutor
from weather_agent import WeatherAgent


# ================= Load ENV =================
load_dotenv(find_dotenv(), override=True)

HOST = "localhost"
PORT = 10002
LOG_LEVEL = "info"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", 60))
VALID_USER = os.getenv("AGENT_USER", "admin")
VALID_PASSWORD = os.getenv("AGENT_PASSWORD", "123456")

app_context: dict[str, Any] = {}

# ================= RSA Key Generation =================
print("🔑 Generating RSA keypair for RS256...")
RSA_KEY = jwk.JWK.generate(kty='RSA', size=2048, kid=str(uuid.uuid4()), use="sig")
PRIVATE_KEY_PEM = RSA_KEY.export_to_pem(private_key=True, password=None)
PUBLIC_KEY_PEM = RSA_KEY.export_to_pem()


# ---------------- JWT Helpers ----------------
def create_jwt_token(username: str) -> str:
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {"sub": username, "exp": expire}
    token = jwt.encode(payload, PRIVATE_KEY_PEM, algorithm="RS256")
    return token


def verify_jwt_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, PUBLIC_KEY_PEM, algorithms=["RS256"])
        return payload.get("sub")
    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid token")


# ---------------- Middleware ----------------
class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/auth/login"):
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse({"detail": "Missing Authorization header"}, status_code=401)

        token = auth_header.split(" ")[1]
        try:
            user = verify_jwt_token(token)
            request.state.user = user
        except ValueError as e:
            return JSONResponse({"detail": str(e)}, status_code=401)

        return await call_next(request)


# ---------------- Auth Route ----------------
async def login(request: Request):
    data = await request.json()
    username = data.get("username")
    password = data.get("password")

    if username == VALID_USER and password == VALID_PASSWORD:
        token = create_jwt_token(username)
        return JSONResponse({"access_token": token, "token_type": "Bearer"})
    return JSONResponse({"detail": "Invalid credentials"}, status_code=401)


# ---------------- Lifespan ----------------
@asynccontextmanager
async def app_lifespan(context: dict[str, Any]):
    print("Initializing Weather Agent...")
    try:
        if GOOGLE_API_KEY is None:
            raise ValueError("GOOGLE_API_KEY is not set")
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)
        weather_agent = WeatherAgent(llm=llm)
        executor = WeatherExecutor(agent=weather_agent)
        context.update({"llm": llm, "weather_agent": weather_agent, "executor": executor})
        yield
    finally:
        print("Cleaning up context...")
        context.clear()


# ---------------- AgentCard ----------------
def get_agent_card(app_url: str):
    capabilities = AgentCapabilities(streaming=True, push_notifications=True)
    skill = AgentSkill(
        id="weather_search",
        name="Search weather information",
        description="Helps with weather information search",
        tags=["weather information"],
        examples=["Please find the weather in Tokyo"]
    )
    return AgentCard(
        name="Weather Agent",
        description="Helps with searching weather information",
        url=app_url,
        version="1.0.0",
        default_input_modes=WeatherExecutor.SUPPORTED_CONTENT_TYPES,
        default_output_modes=WeatherExecutor.SUPPORTED_CONTENT_TYPES,
        capabilities=capabilities,
        skills=[skill],
        security_schemes={
            "httpBearerAuth": SecurityScheme(
                root=HTTPAuthSecurityScheme(
                    scheme="Bearer",
                    bearer_format="JWT (RS256)",
                    description="Use JWT token obtained from /auth/login"
                )
            )
        },
        security=[{"httpBearerAuth": []}],
        supports_authenticated_extended_card=True,
    )


# ---------------- Run Server ----------------
async def run_server_async(host: str, port: int, log_level: str):
    async with app_lifespan(app_context):
        executor = app_context.get("executor")
        httpx_client = None  # will be used by request handler

        # A2A request handler
        push_config_store = InMemoryPushNotificationConfigStore()
        push_sender = BasePushNotificationSender(httpx_client=httpx_client, config_store=push_config_store)
        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
            push_config_store=push_config_store,
            push_sender=push_sender,
        )

        app_url = f"http://{host}:{port}/"
        a2a_server = A2AStarletteApplication(agent_card=get_agent_card(app_url), http_handler=request_handler)
        asgi_app = a2a_server.build()

        # Wrap with Starlette to add auth route + middleware
        app = Starlette()
        app.add_route("/auth/login", login, methods=["POST"])
        app.mount("/", asgi_app)
        app.add_middleware(JWTAuthMiddleware)

        config = uvicorn.Config(app=app, host=host, port=port, log_level=log_level)
        server = uvicorn.Server(config)
        await server.serve()


# ---------------- CLI ----------------
import click

@click.command()
@click.option("--host", default=HOST)
@click.option("--port", default=PORT, type=int)
@click.option("--log-level", default=LOG_LEVEL)
def main(host, port, log_level):
    asyncio.run(run_server_async(host, port, log_level))


if __name__ == "__main__":
    main()
