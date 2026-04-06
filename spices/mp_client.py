"""
Centralized Materials Project API client management.

Replaces hardcoded API keys and global ``mpi`` variables throughout
the original codebase.  Every function that needs the API accepts an
optional ``mp_client`` parameter; if *None*, the module-level default
client is used.
"""

import os
from mp_api.client import MPRester

_default_client = None


def get_client(api_key=None):
    """
    Return an MPRester client instance.

    Resolution order for the API key:
      1. Explicit *api_key* argument
      2. ``MP_API_KEY`` environment variable
      3. ``.env`` file in the current working directory (via python-dotenv)

    The client is cached so repeated calls with the same key reuse
    the connection.

    Parameters
    ----------
    api_key : str or None
        Materials Project API key. If None, resolved from the environment.

    Returns
    -------
    MPRester

    Raises
    ------
    ValueError
        If no API key can be found.
    """
    global _default_client

    if api_key is None:
        api_key = os.environ.get("MP_API_KEY")

    if api_key is None:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get("MP_API_KEY")
        except ImportError:
            pass

    if api_key is None:
        raise ValueError(
            "No Materials Project API key found. Set the MP_API_KEY "
            "environment variable, create a .env file, or pass api_key "
            "explicitly."
        )

    if _default_client is None:
        _default_client = MPRester(api_key)

    return _default_client


def set_api_key(api_key):
    """
    Set (or replace) the default API key for this session.

    Parameters
    ----------
    api_key : str
        Materials Project API key.
    """
    global _default_client
    os.environ["MP_API_KEY"] = api_key
    _default_client = MPRester(api_key)
