import os
if "EXTRA_DLL_SEARCH_PATHS" in os.environ and hasattr(os, "add_dll_directory"):
    for path in os.environ["EXTRA_DLL_SEARCH_PATHS"].split(os.pathsep):
        os.add_dll_directory(path)
import sys
import sysconfig
import subprocess
import importlib.util
import unittest


# A free-threaded (no-GIL) build of Python, e.g. 3.13t/3.14t. On such a build
# the GIL starts *disabled* and is only (permanently) re-enabled at import time
# when a C extension that has NOT declared it can run without the GIL is loaded.
# That is exactly the signal we use below to confirm _xraylib (built by swig with
# -nogil) and xraylib_np (built by cython with freethreading_compatible=True) were
# compiled with free-threading support.
IS_FREETHREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def _import_module_gil_state(modname):
    """Import modname in a *fresh* interpreter and report whether the GIL is
    enabled afterwards. A subprocess is required because re-enabling the GIL is a
    permanent, process-wide side effect: importing one non-declaring module would
    otherwise mask the state of any module imported afterwards in the same process.

    Returns a (gil_enabled, stderr) tuple.
    """
    code = (
        "import os, sys\n"
        "if 'EXTRA_DLL_SEARCH_PATHS' in os.environ and hasattr(os, 'add_dll_directory'):\n"
        "    for path in os.environ['EXTRA_DLL_SEARCH_PATHS'].split(os.pathsep):\n"
        "        os.add_dll_directory(path)\n"
        "import {modname}\n"
        "getter = getattr(sys, '_is_gil_enabled', None)\n"
        "print(getter() if getter is not None else True)\n"
    ).format(modname=modname)
    result = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "failed to import '{}' in a subprocess:\n{}".format(modname, result.stderr)
        )
    return result.stdout.strip() == "True", result.stderr


def _module_available(modname):
    return importlib.util.find_spec(modname) is not None


class TestFreeThreading(unittest.TestCase):

    def _assert_supports_freethreading(self, modname):
        gil_enabled, stderr = _import_module_gil_state(modname)
        self.assertFalse(
            gil_enabled,
            msg="importing '{}' re-enabled the GIL on a free-threaded build, so it "
            "was NOT built with free-threading support:\n{}".format(modname, stderr),
        )

    def _assert_gil_enabled(self, modname):
        gil_enabled, _ = _import_module_gil_state(modname)
        self.assertTrue(
            gil_enabled,
            msg="the GIL is unexpectedly disabled after importing '{}' on a regular "
            "(non free-threaded) Python build".format(modname),
        )

    # --- free-threaded (no-GIL) builds: the modules MUST declare support -----

    @unittest.skipUnless(IS_FREETHREADED, "requires a free-threaded (no-GIL) Python build")
    def test_xraylib_supports_freethreading(self):
        self._assert_supports_freethreading("_xraylib")

    @unittest.skipUnless(IS_FREETHREADED, "requires a free-threaded (no-GIL) Python build")
    @unittest.skipUnless(_module_available("xraylib_np"), "xraylib_np (numpy bindings) not built")
    def test_xraylib_np_supports_freethreading(self):
        self._assert_supports_freethreading("xraylib_np")

    # --- regular (GIL) builds: free-threading is never active -----------------

    @unittest.skipIf(IS_FREETHREADED, "requires a regular (GIL) Python build")
    def test_xraylib_gil_enabled_on_regular_build(self):
        self._assert_gil_enabled("_xraylib")

    @unittest.skipIf(IS_FREETHREADED, "requires a regular (GIL) Python build")
    @unittest.skipUnless(_module_available("xraylib_np"), "xraylib_np (numpy bindings) not built")
    def test_xraylib_np_gil_enabled_on_regular_build(self):
        self._assert_gil_enabled("xraylib_np")


if __name__ == '__main__':
    unittest.main(verbosity=2)
