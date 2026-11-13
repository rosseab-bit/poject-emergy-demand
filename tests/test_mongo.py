"""Tests de la configuración y del cliente compartido de MongoDB (sin red)."""

import threading
from pathlib import Path

import mongomock
import pytest
from pymongo.errors import ServerSelectionTimeoutError

from src import config, mongo

URI = "mongodb+srv://usuario:<db_password>@cluster0.abc.mongodb.net/?appName=app"


@pytest.fixture
def env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Un ``.env`` temporal, ya apuntado desde ``config`` y sin variables del sistema."""
    path = tmp_path / ".env"
    monkeypatch.setattr(config, "ENV_PATH", path)
    mongo.reset_client()  # que get_client() cree uno nuevo desde este .env
    for name in (mongo.ENV_URI, mongo.ENV_PASSWORD, mongo.ENV_DB_NAME):
        monkeypatch.delenv(name, raising=False)
    return path


def write_env(path: Path, password: str = "clave123", extra: str = "") -> None:
    """Escribe un ``.env`` válido."""
    path.write_text(f"URI={URI}\nMONGODB_PASS={password}\n{extra}", encoding="utf-8")


# parse_env_file -----------------------------------------------------------------


def test_parse_env_file_reads_pairs_comments_quotes_and_export(tmp_path: Path) -> None:
    """Ignora comentarios y vacías, acepta ``export`` y quita comillas."""
    path = tmp_path / ".env"
    path.write_text(
        "# comentario\n\nA=1\nexport B = dos \nC=\"con espacios\"\nD='x'\nsin_igual\n"
        "E=a=b\n",
        encoding="utf-8",
    )
    assert mongo.parse_env_file(path) == {
        "A": "1",
        "B": "dos",
        "C": "con espacios",
        "D": "x",
        "E": "a=b",
    }


def test_parse_env_file_missing_or_with_bom(tmp_path: Path) -> None:
    """Un archivo ausente da un diccionario vacío; un BOM no rompe la primera clave."""
    assert mongo.parse_env_file(tmp_path / "no_existe") == {}
    path = tmp_path / ".env"
    path.write_bytes(b"\xef\xbb\xbfA=1\n")
    assert mongo.parse_env_file(path) == {"A": "1"}


# build_uri y describe -------------------------------------------------------------


def test_build_uri_replaces_the_placeholder_with_the_encoded_password() -> None:
    """La clave se codifica para URL: ``@``, ``:``, ``/`` y espacios no rompen la URI."""
    uri = mongo.build_uri(URI, "p@ss:w/rd 1")
    assert "<db_password>" not in uri
    assert "p%40ss%3Aw%2Frd+1" in uri
    assert uri.startswith("mongodb+srv://usuario:p%40ss")


def test_build_uri_without_placeholder_is_unchanged() -> None:
    """Si la URI ya trae la clave, no se toca."""
    plain = "mongodb+srv://u:clave@host.mongodb.net/"
    assert mongo.build_uri(plain, None) == plain
    assert mongo.build_uri(plain, "otra") == plain


@pytest.mark.parametrize("password", [None, ""])
def test_build_uri_with_placeholder_and_no_password_fails(password: str | None) -> None:
    """Con marcador y sin clave hay un error claro que nombra la variable."""
    with pytest.raises(mongo.MongoConfigError, match="MONGODB_PASS"):
        mongo.build_uri(URI, password)


def test_describe_hides_the_password() -> None:
    """La descripción muestra usuario y host, nunca la clave."""
    uri = mongo.build_uri(URI, "secreta")
    assert mongo.describe(uri) == "mongodb+srv://usuario@cluster0.abc.mongodb.net"
    assert "secreta" not in mongo.describe(uri)
    assert mongo.describe("mongodb://localhost:27017") == "mongodb://localhost"


# load_settings --------------------------------------------------------------------


def test_load_settings_reads_the_env_file(env_file: Path) -> None:
    """Toma la URI y la clave del ``.env`` y usa el nombre de base por defecto."""
    write_env(env_file)
    settings = mongo.load_settings()
    assert settings.db_name == config.MONGO_DB_NAME
    assert "clave123" in settings.uri and "<" not in settings.uri


def test_load_settings_hides_the_uri_in_repr(env_file: Path) -> None:
    """El ``repr`` no deja ver la URI con la clave."""
    write_env(env_file, password="muysecreta")
    assert "muysecreta" not in repr(mongo.load_settings())


def test_load_settings_custom_db_name(env_file: Path) -> None:
    """``MONGODB_DB`` cambia el nombre de la base."""
    write_env(env_file, extra="MONGODB_DB=otra_base\n")
    assert mongo.load_settings().db_name == "otra_base"


def test_system_environment_wins_over_the_file(env_file: Path) -> None:
    """Una variable del entorno pisa lo que diga el archivo."""
    write_env(env_file, password="del_archivo")
    settings = mongo.load_settings(environ={"MONGODB_PASS": "del_entorno"})
    assert "del_entorno" in settings.uri and "del_archivo" not in settings.uri


def test_load_settings_without_env_file_explains_what_to_do(env_file: Path) -> None:
    """Sin ``.env`` el error indica copiar ``.env.example`` y dónde."""
    with pytest.raises(mongo.MongoConfigError, match=r"\.env\.example") as error:
        mongo.load_settings()
    assert str(env_file) in str(error.value)


def test_load_settings_missing_password(env_file: Path) -> None:
    """Con URI con marcador y sin clave, el error nombra ``MONGODB_PASS``."""
    env_file.write_text(f"URI={URI}\n", encoding="utf-8")
    with pytest.raises(mongo.MongoConfigError, match="MONGODB_PASS"):
        mongo.load_settings()


# Cliente compartido ---------------------------------------------------------------


def test_use_client_and_get_database(mongo_db: mongomock.Database) -> None:
    """``get_database`` devuelve la base del cliente instalado."""
    assert mongo.get_database().name == "demanda_test"
    mongo.get_database()["x"].insert_one({"a": 1})
    assert mongo_db["x"].count_documents({}) == 1


def test_get_client_is_created_once_even_from_many_threads(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Muchos hilos obtienen el mismo cliente y se crea uno solo."""
    write_env(env_file)
    created: list[dict] = []

    def fake_client(uri: str, **kwargs: object) -> mongomock.MongoClient:
        created.append({"uri": uri, **kwargs})
        return mongomock.MongoClient()

    monkeypatch.setattr(mongo, "MongoClient", fake_client)
    seen: list[object] = []
    threads = [
        threading.Thread(target=lambda: seen.append(mongo.get_client()))
        for _ in range(20)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(created) == 1
    assert len({id(client) for client in seen}) == 1
    assert created[0]["serverSelectionTimeoutMS"] == config.MONGO_TIMEOUT_MS
    assert created[0]["appname"] == mongo.APP_NAME


def test_reset_client_makes_the_next_call_create_a_new_one(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Después de ``reset_client`` se vuelve a crear el cliente."""
    write_env(env_file)
    monkeypatch.setattr(mongo, "MongoClient", lambda *a, **k: mongomock.MongoClient())
    first = mongo.get_client()
    mongo.reset_client()
    assert mongo.get_client() is not first


def test_get_client_without_configuration_fails_clearly(env_file: Path) -> None:
    """Sin ``.env`` no se intenta conectar: hay un error de configuración."""
    with pytest.raises(mongo.MongoConfigError):
        mongo.get_client()


# check_connection -----------------------------------------------------------------


def test_check_connection_succeeds_with_a_working_client(
    mongo_db: mongomock.Database,
) -> None:
    """Con un cliente que responde, no lanza nada."""
    mongo.check_connection()


def test_check_connection_failure_hides_the_password(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Si el servidor falla, el error es claro y no incluye la clave ni la URI."""
    write_env(env_file, password="muysecreta")
    settings = mongo.load_settings()

    class Broken:
        class admin:
            @staticmethod
            def command(name: str) -> None:
                raise ServerSelectionTimeoutError(
                    f"no responde {settings.uri} con muysecreta"
                )

        def close(self) -> None:
            """Nada que cerrar."""

    mongo.use_client(Broken())  # type: ignore[arg-type]
    with pytest.raises(mongo.MongoUnavailableError) as error:
        mongo.check_connection()
    message = str(error.value)
    assert "ServerSelectionTimeoutError" in message
    assert "muysecreta" not in message and "mongodb+srv" not in message


def test_redact_without_configuration_returns_the_message(env_file: Path) -> None:
    """Sin configuración no hay nada que ocultar y el mensaje sale tal cual."""
    assert mongo._redact("mensaje") == "mensaje"


def test_redact_with_a_password_free_uri(
    env_file: Path,
) -> None:
    """Una URI sin clave no rompe la limpieza del mensaje."""
    env_file.write_text("URI=mongodb://localhost:27017\n", encoding="utf-8")
    assert mongo._redact("falló localhost") == "falló localhost"


def test_reset_client_without_a_client_is_harmless() -> None:
    """Reiniciar dos veces seguidas no falla: no hay nada que cerrar."""
    mongo.reset_client()
    mongo.reset_client()
    assert mongo._Connection.client is None


def test_restore_client_brings_back_the_previous_state(
    mongo_db: mongomock.Database,
) -> None:
    """``use_client`` devuelve el estado anterior y ``restore_client`` lo recupera."""
    other = mongomock.MongoClient()
    previous = mongo.use_client(other, "otra")
    assert mongo.get_database().name == "otra"
    mongo.restore_client(previous)
    assert mongo.get_database().name == "demanda_test"
