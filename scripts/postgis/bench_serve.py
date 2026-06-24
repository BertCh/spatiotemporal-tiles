#!/usr/bin/env python3
"""Concurrent HTTP latency driver for the serve benchmark.

Fires the given tile paths at a base URL with N concurrent workers (persistent
keep-alive connections) and reports latency percentiles, throughput, the
server-side generation time (from the `x-stt-gen-micros` header, dynamic server
only) and average tile size.

Usage: bench_serve.py <base_url> <urls_file> <concurrency> <label>
"""
import http.client
import statistics
import sys
import threading
import time
from urllib.parse import urlparse


def main() -> None:
    base, listfile, conc, label = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    urls = [ln.strip() for ln in open(listfile) if ln.strip()]
    u = urlparse(base)
    host, port = u.hostname, u.port or 80

    lat_ms, gen_ms, statuses = [], [], {}
    total_bytes = 0
    errors = [0]
    idx = [0]
    lock = threading.Lock()

    def do_request(conn, path):
        t0 = time.perf_counter()
        conn.request("GET", path)
        r = conn.getresponse()
        body = r.read()
        dt = (time.perf_counter() - t0) * 1000.0
        return r.status, r.getheader("x-stt-gen-micros"), len(body), dt

    def worker():
        nonlocal total_bytes
        conn = http.client.HTTPConnection(host, port, timeout=60)
        while True:
            with lock:
                i = idx[0]
                idx[0] += 1
            if i >= len(urls):
                break
            path = urls[i]
            status = g = nbytes = dt = None
            # Servers recycle keep-alive connections; reconnect + retry once.
            for attempt in range(2):
                try:
                    status, g, nbytes, dt = do_request(conn, path)
                    break
                except (http.client.HTTPException, OSError, ConnectionError):
                    try:
                        conn.close()
                    except OSError:
                        pass
                    conn = http.client.HTTPConnection(host, port, timeout=60)
            with lock:
                if status is None:
                    errors[0] += 1
                    continue
                lat_ms.append(dt)
                statuses[status] = statuses.get(status, 0) + 1
                total_bytes += nbytes
                if g:
                    gen_ms.append(int(g) / 1000.0)
        conn.close()

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(conc)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    def pct(a, p):
        if not a:
            return 0.0
        a = sorted(a)
        return a[min(len(a) - 1, int(round(len(a) * p / 100.0)))]

    n = len(lat_ms)
    print(f"[{label}]")
    print(
        f"  {n} reqs, conc={conc}, {wall:.2f}s wall, {n / wall:.0f} req/s, "
        f"statuses={statuses}, conn_errors={errors[0]}"
    )
    print(
        f"  latency ms:  p50={pct(lat_ms,50):.2f}  p95={pct(lat_ms,95):.2f}  "
        f"p99={pct(lat_ms,99):.2f}  max={max(lat_ms):.2f}  mean={statistics.mean(lat_ms):.2f}"
    )
    if gen_ms:
        print(
            f"  server gen:  p50={pct(gen_ms,50):.2f}  p95={pct(gen_ms,95):.2f}  "
            f"(pure PostGIS->tile compute, ms)"
        )
    print(f"  avg tile:    {total_bytes / max(n,1):.0f} bytes")


if __name__ == "__main__":
    main()
