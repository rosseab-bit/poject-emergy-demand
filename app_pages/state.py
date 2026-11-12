"""Estado de la interfaz: corrida elegida y semilla de la próxima corrida.

Solo guarda preferencias de pantalla. Los datos y resultados viven en las
bases y se leen con ``src.queries``.
"""

import streamlit as st

RUN_KEY = "run_selector"
PAGE_KEY = "page"
SEED_KEY = "seed"
PENDING_RUN_KEY = "_pending_run"
FLASH_KEY = "_flash"


def current_run_id() -> str | None:
    """Devuelve el ``run_id`` elegido en la barra lateral, o ``None``."""
    return st.session_state.get(RUN_KEY)


def current_seed(default: int) -> int:
    """Devuelve la semilla escrita en la barra lateral (para la próxima corrida)."""
    return int(st.session_state.get(SEED_KEY, default))


def select_run(run_id: str) -> None:
    """Pide que la barra lateral pase a mostrar ``run_id`` y vuelve a ejecutar.

    Streamlit no deja cambiar un widget ya creado en la misma ejecución, así que
    el pedido queda pendiente y lo aplica ``apply_pending_run`` al empezar.
    """
    st.session_state[PENDING_RUN_KEY] = run_id
    st.rerun()


def apply_pending_run() -> None:
    """Aplica la corrida pedida con ``select_run``; llamar antes del selector."""
    pending = st.session_state.pop(PENDING_RUN_KEY, None)
    if pending is not None:
        st.session_state[RUN_KEY] = pending


def flash(message: str) -> None:
    """Deja un aviso para mostrar en la próxima ejecución (sobrevive al rerun)."""
    st.session_state[FLASH_KEY] = message


def show_flash() -> None:
    """Muestra el aviso pendiente, si lo hay, y lo borra."""
    message = st.session_state.pop(FLASH_KEY, None)
    if message:
        st.success(message)
