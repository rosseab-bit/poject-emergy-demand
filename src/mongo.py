"""Conexión a MongoDB (Atlas): credenciales desde ``.env`` y un cliente compartido.

MongoDB guarda los logs, las corridas, los resultados, la configuración y las
predicciones (ver ``docs/plan_proyecto.md``, sección 12). Las credenciales viven
en un archivo ``.env`` que no se sube a Git:

``URI``
    URI de conexión de Atlas, con el marcador ``<db_password>`` donde va la clave.
``MONGODB_PASS``
    Clave del usuario de la base; reemplaza al marcador de la URI.
``MONGODB_DB``
    Nombre de la base (opcional; por defecto ``config.MONGO_DB_NAME``).

Las variables de entorno del sistema tienen prioridad sobre el archivo.

A diferencia de SQLite, un ``MongoClient`` se comparte entre hilos y conviene
crearlo una sola vez: abrir uno por operación costaría ~0,6 s cada vez.
Ningún mensaje de error incluye la clave ni la URI completa.
"""

import logging
import os
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote_plus, unquote_plus, urlsplit

from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import PyMongoError

from src import config

log = logging.getLogger(__name__)

ENV_URI = "URI"
ENV_PASSWORD = "MONGODB_PASS"
ENV_DB_NAME = "MONGODB_DB"

# Colecciones de la base (las crea ``init_mongo`` al arrancar).
LOGS = "logs"
RUNS = "runs"
RESULTS = "resultados_modelos"
CONFIGURATION = "configuracion"
PREDICTIONS = "predicciones"

PASSWORD_PLACEHOLDER = re.compile(r"<[^<>]*>")
APP_NAME = "tp-demanda-electrica"


class MongoConfigError(RuntimeError):
    """Falta o está mal la configuración de MongoDB (``.env``)."""


class MongoUnavailableError(RuntimeError):
    """No se pudo hablar con el servidor de MongoDB."""


@dataclass(frozen=True)
class MongoSettings:
    """Configuración de conexión. La URI lleva la clave, por eso no se imprime."""

    uri: str = field(repr=False)
    db_name: str


def parse_env_file(path: Path) -> dict[str, str]:
    """Lee un archivo ``.env`` simple de líneas ``CLAVE=valor``.

    Ignora líneas vacías y comentarios, acepta el prefijo ``export`` y quita
    comillas simples o dobles que rodeen el valor.

    Args:
        path: ruta del archivo; si no existe, devuelve un diccionario vacío.

    Returns:
        Las variables definidas en el archivo.
    """
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = _unquote(value.strip())
    return values


def _unquote(value: str) -> str:
    """Quita un par de comillas que rodee el valor."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def build_uri(uri: str, password: str | None) -> str:
    """Reemplaza el marcador ``<...>`` de la URI por la clave codificada.

    Args:
        uri: URI de conexión, con o sin marcador de clave.
        password: clave del usuario; se codifica para URL (``@``, ``:``, ``/``).

    Returns:
        La URI lista para conectar.

    Raises:
        MongoConfigError: si la URI tiene marcador y no hay clave.
    """
    if not PASSWORD_PLACEHOLDER.search(uri):
        return uri
    if not password:
        raise MongoConfigError(
            f"La URI tiene un marcador de clave y falta {ENV_PASSWORD} en el .env."
        )
    encoded = quote_plus(password)
    return PASSWORD_PLACEHOLDER.sub(lambda _match: encoded, uri, count=1)


def describe(uri: str) -> str:
    """Describe un destino sin secretos: ``esquema://usuario@host``."""
    parts = urlsplit(uri)
    user = f"{parts.username}@" if parts.username else ""
    return f"{parts.scheme}://{user}{parts.hostname or '?'}"


def load_settings(
    env_path: Path | None = None, environ: Mapping[str, str] | None = None
) -> MongoSettings:
    """Arma la configuración desde el ``.env`` y las variables de entorno.

    Args:
        env_path: ruta del ``.env``; por defecto ``config.ENV_PATH``.
        environ: variables de entorno (para pruebas); por defecto, las del
            sistema. Tienen prioridad sobre el archivo.

    Returns:
        La configuración, con la clave ya incluida en la URI.

    Raises:
        MongoConfigError: si falta el archivo o una variable obligatoria.
    """
    path = env_path or config.ENV_PATH
    values = {
        **parse_env_file(path),
        **{
            key: value
            for key, value in (environ if environ is not None else os.environ).items()
            if key in (ENV_URI, ENV_PASSWORD, ENV_DB_NAME)
        },
    }
    uri = values.get(ENV_URI, "").strip()
    if not uri:
        raise MongoConfigError(
            f"Falta la variable {ENV_URI} de MongoDB. Copiá .env.example como .env "
            f"({path}) y completá tus datos."
        )
    return MongoSettings(
        uri=build_uri(uri, values.get(ENV_PASSWORD)),
        db_name=values.get(ENV_DB_NAME, "").strip() or config.MONGO_DB_NAME,
    )


class _Connection:
    """Estado compartido: el cliente y el nombre de la base."""

    client: MongoClient | None = None
    db_name: str = config.MONGO_DB_NAME
    lock = threading.Lock()


def get_client() -> MongoClient:
    """Devuelve el ``MongoClient`` compartido, creándolo la primera vez.

    Es seguro entre hilos: un candado evita crear dos clientes a la vez.

    Raises:
        MongoConfigError: si no hay configuración válida en el ``.env``.
    """
    if _Connection.client is None:
        with _Connection.lock:
            if _Connection.client is None:
                settings = load_settings()
                _Connection.client = MongoClient(
                    settings.uri,
                    serverSelectionTimeoutMS=config.MONGO_TIMEOUT_MS,
                    connectTimeoutMS=config.MONGO_TIMEOUT_MS,
                    appname=APP_NAME,
                )
                _Connection.db_name = settings.db_name
                log.debug("Cliente MongoDB creado: %s", describe(settings.uri))
    return _Connection.client


def get_database() -> Database:
    """Devuelve la base del proyecto sobre el cliente compartido."""
    client = get_client()
    return client[_Connection.db_name]


def use_client(client: MongoClient, db_name: str | None = None) -> None:
    """Fija el cliente compartido (para pruebas, por ejemplo con ``mongomock``)."""
    with _Connection.lock:
        _Connection.client = client
        _Connection.db_name = db_name or config.MONGO_DB_NAME


def reset_client() -> None:
    """Cierra y olvida el cliente compartido; el próximo uso crea uno nuevo."""
    with _Connection.lock:
        if _Connection.client is not None:
            _Connection.client.close()
        _Connection.client = None
        _Connection.db_name = config.MONGO_DB_NAME


def check_connection() -> None:
    """Hace un ``ping`` al servidor para comprobar que se puede conectar.

    Raises:
        MongoConfigError: si falta la configuración.
        MongoUnavailableError: si el servidor no responde o rechaza las
            credenciales. El mensaje no incluye la clave.
    """
    client = get_client()
    try:
        client.admin.command("ping")
    except PyMongoError as error:
        raise MongoUnavailableError(
            f"No se pudo conectar a MongoDB: {type(error).__name__}: "
            f"{_redact(str(error))}"
        ) from None


def _redact(message: str) -> str:
    """Quita del mensaje la clave y la URI completas, si aparecieran."""
    try:
        settings = load_settings()
    except MongoConfigError:
        return message
    secrets = [settings.uri, *_password_variants(settings.uri)]
    for secret in sorted(filter(None, secrets), key=len, reverse=True):
        message = message.replace(secret, "***")
    return message


def _password_variants(uri: str) -> list[str]:
    """La clave tal como puede aparecer: codificada y sin codificar."""
    password = urlsplit(uri).password
    if not password:
        return []
    return [password, unquote_plus(password)]
