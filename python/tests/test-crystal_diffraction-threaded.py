import os
if "EXTRA_DLL_SEARCH_PATHS" in os.environ and hasattr(os, "add_dll_directory"):
    for path in os.environ["EXTRA_DLL_SEARCH_PATHS"].split(os.pathsep):
        os.add_dll_directory(path)
import threading
import unittest
import xraylib


# Hammer the process-global crystal array (the c_array=None path) from several
# threads at once. That global is serialized by an internal lock now that the
# bindings declare free-threading support; on a 3.13t/3.14t build these run
# truly in parallel and exercise it directly.
N_THREADS = 8
ADDS_PER_THREAD = 40  # 8 * 40 + 38 built-ins = 358 < CRYSTALARRAY_MAX (512)


class TestCrystalDiffractionThreaded(unittest.TestCase):

    def test_concurrent_readers(self):
        errors = []
        barrier = threading.Barrier(N_THREADS)

        def worker():
            barrier.wait()  # release all threads at once to maximize overlap
            try:
                for _ in range(200):
                    names = xraylib.Crystal_GetCrystalsList()
                    self.assertGreaterEqual(len(names), 38)
                    for name in names:
                        cs = xraylib.Crystal_GetCrystal(name)
                        self.assertEqual(cs['name'], name)
            except Exception as e:  # noqa: BLE001 - report, don't swallow
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [], msg="reader threads raised: {}".format(errors))

    def test_concurrent_readers_and_writers(self):
        # Adders mutate/insertion-sort (and may grow) the array while readers
        # bsearch/copy it -- the actual race the lock protects against.
        seed = xraylib.Crystal_GetCrystal("Diamond")
        start = len(xraylib.Crystal_GetCrystalsList())
        self.assertLess(start + N_THREADS * ADDS_PER_THREAD,
                        xraylib.CRYSTALARRAY_MAX)
        errors = []
        barrier = threading.Barrier(N_THREADS)

        def worker(tid):
            barrier.wait()
            try:
                for i in range(ADDS_PER_THREAD):
                    # concurrent readers on the shared global
                    names = xraylib.Crystal_GetCrystalsList()
                    probe = names[i % len(names)]
                    cs = xraylib.Crystal_GetCrystal(probe)
                    self.assertEqual(cs['name'], probe)
                    # concurrent writer on the shared global (unique name)
                    cpy = xraylib.Crystal_MakeCopy(seed)
                    cpy['name'] = "T{}-{}".format(tid, i)
                    xraylib.Crystal_AddCrystal(cpy)
            except Exception as e:  # noqa: BLE001 - report, don't swallow
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,))
                   for t in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], msg="worker threads raised: {}".format(errors))
        # every add was a unique name, none hit the cap: no lost/duplicated entries
        final = xraylib.Crystal_GetCrystalsList()
        self.assertEqual(len(final), start + N_THREADS * ADDS_PER_THREAD)
        self.assertEqual(len(set(final)), len(final), msg="duplicate crystal names")
        for name in final:
            self.assertEqual(xraylib.Crystal_GetCrystal(name)['name'], name)


if __name__ == '__main__':
    unittest.main(verbosity=2)
