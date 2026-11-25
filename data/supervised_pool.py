# /data/supervised_pool.py

"""Workers under a deadline for the long archive jobs: video verification (build_manifests.py) and
the DICOM header scan (scan_acquisition.py).

Each task runs in a worker process with a deadline. A worker that overruns it is killed (a thread or
a cancelled future cannot stop native decoding or network-file-system I/O; a process can be killed),
a worker that dies is noticed through its process sentinel, and either one is replaced; the task is
then retried a bounded number of times. A task still failing after its attempts comes back as a
`Failure`, never as a result, so the caller can leave it uncached for a later run to retry.
Log lines name tasks by index only: the items (file paths) can hold patient identifiers.
"""

import heapq
import itertools
import multiprocessing.process
import queue
import signal
import threading
import time
from multiprocessing import connection, get_context
from typing import NamedTuple

TIMEOUT, CRASH, ERROR = "timeout", "crash", "error"
KINDS = (TIMEOUT, CRASH, ERROR)

# A new worker must import its modules and report ready within this time (or the task deadline if longer).
STARTUP_TIMEOUT = 120.0

# Seconds a killed worker gets to exit before it is left behind as stuck (blocked in the kernel).
KILL_WAIT = 5.0


class Failure(NamedTuple):
    """A task left without a result after `attempts` attempts. `kind` is that of the last attempt:
    `timeout` (no result before the deadline; the worker was killed), `crash` (the worker died) or
    `error` (the task raised `detail`, an exception type name)."""

    kind: str
    attempts: int
    detail: str | None = None


class PoolBroken(RuntimeError):
    """The work stops: attempts keep failing with no result in between (storage down?), workers fail
    to start, killed workers cannot be reaped, or (`thread_map`) nothing completes for too long."""


def _serve(conn, fn, init):
    """Worker loop: report ready, then answer one task per message until the parent says stop."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C stops the parent, which then kills the workers.
    if init is not None:
        init()
    conn.send(None)
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):  # the parent is gone
            return
        if msg is None:
            return
        key, item = msg
        try:
            out = (key, True, fn(item))
        except Exception as e:
            out = (key, False, type(e).__name__)
        conn.send(out)


class _Worker:
    def __init__(self, proc, conn, deadline):
        self.proc, self.conn, self.deadline = proc, conn, deadline
        self.ready, self.task = False, None  # task: (index, attempt)


class SupervisedPool:
    """`workers` processes, each running `fn` on one item at a time under a deadline of `timeout` s.

    A task that times out, crashes its worker or raises is tried again, up to `retries` more times;
    so is a task whose result makes `retry_result(result)` true (e.g. a decode failure that a storage
    hiccup can cause). A retry waits `retry_delay` x the attempts made so far, while other tasks run.
    `init` runs once in each new worker. `fn` and `init` must be picklable (module-level functions or
    partials of them). Use it as a context manager: leaving it stops the workers.

    The pool stops (PoolBroken) after `max_streak` failed attempts in a row with no result in between
    (default: every worker failing twice plus one task's attempts), when workers keep dying before
    they are ready, or when more than `workers` killed processes cannot be reaped.
    """

    def __init__(self, fn, workers, timeout, retries=2, retry_result=None, retry_delay=0.0, init=None,
                 log=print, max_streak=None):
        if workers < 1 or not timeout > 0 or retries < 0:
            raise ValueError("SupervisedPool needs workers >= 1, timeout > 0 and retries >= 0.")
        self.fn, self.n, self.timeout, self.retries = fn, workers, float(timeout), retries
        self.retry_result, self.retry_delay, self.init, self.log = retry_result, retry_delay, init, log
        self.max_streak = max_streak or 2 * workers + retries + 1
        self.ctx = get_context("spawn")  # the parent already runs Arrow / decoder threads: never fork it
        self.workers, self.stuck = [], []
        self.streak = self.start_failures = 0
        self.broken = None
        self.counts = {TIMEOUT: 0, CRASH: 0, ERROR: 0, "retried_results": 0, "recovered": 0, "workers_replaced": 0}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _start(self):
        parent, child = self.ctx.Pipe()
        proc = self.ctx.Process(target=_serve, args=(child, self.fn, self.init), daemon=True)
        proc.start()
        child.close()
        return _Worker(proc, parent, time.monotonic() + max(STARTUP_TIMEOUT, self.timeout))

    def _kill(self, w):
        try:
            w.proc.kill()
        except OSError:
            pass
        w.proc.join(KILL_WAIT)
        w.conn.close()
        if w.proc.is_alive():
            # Blocked in uninterruptible I/O: left behind, and kept out of multiprocessing's exit-time
            # join so this process can still exit.
            getattr(multiprocessing.process, "_children", set()).discard(w.proc)
            self.stuck.append(w.proc)
        self.stuck = [p for p in self.stuck if p.is_alive()]

    def close(self):
        """Stop every worker: idle ones exit at once; any still busy after 2 s is killed."""
        for w in self.workers:
            try:
                w.conn.send(None)
            except OSError:
                pass
        end = time.monotonic() + 2.0
        for w in self.workers:
            w.proc.join(max(0.0, end - time.monotonic()))
            if w.proc.is_alive():
                self._kill(w)
            else:
                w.conn.close()
        self.workers = []

    def _requeue(self, retry, seq, i, attempt):
        heapq.heappush(retry, (time.monotonic() + self.retry_delay * attempt, next(seq), i, attempt + 1))

    def _fail(self, retry, seq, out, i, attempt, kind, detail=None):
        self.counts[kind] += 1
        self.streak += 1
        last = attempt > self.retries
        what = {TIMEOUT: f"no result within {self.timeout:g} s, worker killed", CRASH: "worker died",
                ERROR: f"raised {detail}"}[kind]
        self.log(f"task {i}: {what} (attempt {attempt} of {self.retries + 1}); "
                 + ("left unresolved." if last else "will be retried."))
        if last:
            out.append((i, Failure(kind, attempt, detail)))
        else:
            self._requeue(retry, seq, i, attempt)
        if self.streak >= self.max_streak and self.broken is None:
            counts = {k: self.counts[k] for k in KINDS}
            self.broken = (f"{self.streak} task attempts failed in a row with no result in between {counts}: "
                           "the storage or the workers are failing.")

    def _replace(self, k, kind, retry, seq, out):
        """Kill worker `k` (dead, overdue or failing to start), fail its task, start a replacement."""
        w = self.workers[k]
        self._kill(w)
        if w.task is not None:
            self._fail(retry, seq, out, *w.task, kind)
        elif not w.ready:
            self.start_failures += 1
            if self.start_failures >= max(3, self.n) and self.broken is None:
                self.broken = f"{self.start_failures} worker processes in a row failed to start (see the errors above)."
        if len(self.stuck) > self.n and self.broken is None:
            self.broken = f"{len(self.stuck)} killed workers cannot be reaped (blocked in I/O: storage down?)."
        self.counts["workers_replaced"] += 1
        self.workers[k] = self._start()

    def _drain(self, w, retry, seq, out):
        """Handle every message waiting from `w`; False once its pipe is closed (the worker is gone)."""
        try:
            while w.conn.poll():
                msg = w.conn.recv()
                if msg is None:
                    w.ready, self.start_failures = True, 0
                    continue
                key, ok, value = msg
                if w.task is None or w.task[0] != key:
                    continue
                i, attempt = w.task
                w.task = None
                if not ok:
                    self._fail(retry, seq, out, i, attempt, ERROR, value)
                    continue
                self.streak = 0
                again = self.retry_result is not None and bool(self.retry_result(value))
                if again and attempt <= self.retries:
                    self.counts["retried_results"] += 1
                    self._requeue(retry, seq, i, attempt)
                    continue
                if attempt > 1 and not again:
                    self.counts["recovered"] += 1
                out.append((i, value))
        except (EOFError, OSError):
            return False
        return True

    def imap_unordered(self, items):
        """Yield (i, outcome) as each of `items` (a sequence) is done: `fn(items[i])`, or a `Failure`
        once its attempts are spent. Raises PoolBroken (see the class), after yielding what is done."""
        fresh, retry, seq, left = iter(range(len(items))), [], itertools.count(), len(items)
        while left and len(self.workers) < self.n:
            self.workers.append(self._start())
        while left:
            now = time.monotonic()
            for w in self.workers:
                if w.ready and w.task is None:
                    if retry and retry[0][0] <= now:
                        task = heapq.heappop(retry)[2:]
                    else:
                        i = next(fresh, None)
                        if i is None:
                            break
                        task = (i, 1)
                    w.task, w.deadline = task, now + self.timeout
                    try:
                        w.conn.send((task[0], items[task[0]]))
                    except OSError:  # the worker just died: its sentinel reports it below
                        pass
            wake = [w.deadline for w in self.workers if w.task is not None or not w.ready]
            wake += [retry[0][0]] if retry else []
            wait = min(max(0.0, min(wake, default=now + 1.0) - time.monotonic()), 1.0)
            owner = {}
            for k, w in enumerate(self.workers):
                owner[w.conn], owner[w.proc.sentinel] = k, k
            out = []
            for k in sorted({owner[o] for o in connection.wait(list(owner), wait)}):
                w = self.workers[k]
                if not self._drain(w, retry, seq, out) or not w.proc.is_alive():
                    self._replace(k, CRASH, retry, seq, out)
            now = time.monotonic()
            for k, w in enumerate(self.workers):
                if (w.ready and w.task is None) or now <= w.deadline or w.conn.poll():
                    continue  # idle, on time, or its answer arrived just now (read next round)
                self._replace(k, TIMEOUT, retry, seq, out)
            for done in out:
                left -= 1
                yield done
            if self.broken:
                raise PoolBroken(self.broken)


class InlinePool:
    """`SupervisedPool`'s interface run in this process, for one worker: no deadline and no isolation
    (an exception propagates). A result for which `retry_result` is true is retried at once."""

    def __init__(self, fn, retries=2, retry_result=None):
        self.fn, self.retries, self.retry_result = fn, retries, retry_result
        self.counts = {TIMEOUT: 0, CRASH: 0, ERROR: 0, "retried_results": 0, "recovered": 0, "workers_replaced": 0}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def imap_unordered(self, items):
        for i, item in enumerate(items):
            for attempt in range(1, self.retries + 2):
                value = self.fn(item)
                again = self.retry_result is not None and bool(self.retry_result(value))
                if not again or attempt > self.retries:
                    break
                self.counts["retried_results"] += 1
            if attempt > 1 and not again:
                self.counts["recovered"] += 1
            yield i, value


def thread_map(fn, items, threads, timeout):
    """[fn(x) for x in items] on `threads` daemon threads, for I/O that releases the GIL (stat).

    Raises PoolBroken when no item completes for `timeout` seconds. A thread blocked in network I/O
    cannot be stopped, but the caller is no longer kept waiting (daemon threads do not hold up exit).
    """
    items = list(items)
    results = [None] * len(items)
    if not items:
        return results
    todo, done, stop = iter(range(len(items))), queue.SimpleQueue(), threading.Event()
    lock = threading.Lock()

    def run():
        while not stop.is_set():
            with lock:
                i = next(todo, None)
            if i is None:
                return
            try:
                results[i] = fn(items[i])
            except BaseException as e:  # handed to the caller
                done.put(e)
                return
            done.put(None)

    for _ in range(max(1, min(threads, len(items)))):
        threading.Thread(target=run, daemon=True).start()
    try:
        for _ in range(len(items)):
            try:
                error = done.get(timeout=timeout)
            except queue.Empty:
                raise PoolBroken(f"no progress for {timeout:g} s ({len(items)} items on {threads} threads): "
                                 "the storage is not answering.") from None
            if error is not None:
                raise error
    finally:
        stop.set()
    return results
