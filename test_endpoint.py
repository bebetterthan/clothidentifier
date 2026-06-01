#!/usr/bin/env python3
"""
test_endpoint.py — Test Clothing Classifier GCE endpoint.

Usage:
    python test_endpoint.py --url http://[EXTERNAL_IP] --image test.jpg --runs 10
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import requests


# ----------------------------------------------------------------
# Individual test functions
# ----------------------------------------------------------------

def test_health(base_url: str) -> bool:
    """Test /health endpoint and print model status."""
    print(f"\n{'='*55}")
    print("TEST: GET /health")
    print("="*55)
    try:
        resp = requests.get(f"{base_url}/health", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        print(f"  HTTP status    : {resp.status_code}")
        print(f"  status         : {data.get('status')}")
        print(f"  model_loaded   : {data.get('model_loaded')}")
        print(f"  inference_count: {data.get('inference_count')}")
        print(f"  avg_latency_ms : {data.get('avg_latency_ms')}")
        if data.get("model_loaded"):
            print("  [PASS] Model loaded and healthy.")
            return True
        else:
            print("  [FAIL] model_loaded is False.")
            return False
    except Exception as exc:
        print(f"  [FAIL] {exc}")
        return False


def test_predict(base_url: str, image_path: str) -> bool:
    """Test /predict with a real image file and print classification result."""
    print(f"\n{'='*55}")
    print(f"TEST: POST /predict  ({image_path})")
    print("="*55)

    path = Path(image_path)
    if not path.exists():
        print(f"  [FAIL] Image not found: {image_path}")
        return False

    with open(path, "rb") as f:
        files = {"image": (path.name, f, "image/jpeg")}
        try:
            resp = requests.post(f"{base_url}/predict", files=files, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            body = ""
            if hasattr(exc, "response") and exc.response is not None:
                body = exc.response.text[:400]
            print(f"  [FAIL] {exc}")
            if body:
                print(f"  Response body: {body}")
            return False

    print(f"  HTTP status         : {resp.status_code}")
    print(f"  predicted_class     : {data.get('predicted_class')}")
    print(f"  confidence          : {data.get('confidence'):.5f}")
    print(f"  is_confident        : {data.get('is_confident')}")
    print(f"  threshold           : {data.get('threshold')}")
    print(f"  inference_time_ms   : {data.get('inference_time_ms')}")
    print(f"  image_received_size : {data.get('image_received_size')}")
    print("  probabilities (sorted):")
    for cls, prob in sorted(
        data.get("probabilities", {}).items(), key=lambda x: -x[1]
    ):
        bar = "█" * int(prob * 30)
        print(f"    {cls:<25} {prob:.5f}  {bar}")

    print("  [PASS]")
    return True


def test_latency(base_url: str, image_path: str, n: int = 10) -> bool:
    """Send N predict requests and report RTT latency statistics."""
    print(f"\n{'='*55}")
    print(f"TEST: latency  ({n} runs)")
    print("="*55)

    path = Path(image_path)
    if not path.exists():
        print(f"  [FAIL] Image not found: {image_path}")
        return False

    raw   = path.read_bytes()
    rtts: list[float]       = []
    server_lats: list[float] = []

    for i in range(n):
        files = {"image": (path.name, raw, "image/jpeg")}
        try:
            t0   = time.perf_counter()
            resp = requests.post(f"{base_url}/predict", files=files, timeout=15)
            rtt  = round((time.perf_counter() - t0) * 1000, 1)
            resp.raise_for_status()
            server_ms = float(resp.json().get("inference_time_ms", 0))
            rtts.append(rtt)
            server_lats.append(server_ms)
            print(f"  Run {i+1:>2}/{n}  RTT={rtt:6.1f}ms  server={server_ms:5.1f}ms")
        except Exception as exc:
            print(f"  Run {i+1:>2}/{n}  [FAIL] {exc}")

    if not rtts:
        print("  [FAIL] All requests failed.")
        return False

    p95_idx = max(0, int(len(rtts) * 0.95) - 1)
    print(f"\n  Results ({len(rtts)}/{n} successful):")
    print(f"  Avg RTT    : {statistics.mean(rtts):.2f} ms")
    print(f"  Min RTT    : {min(rtts):.2f} ms")
    print(f"  Max RTT    : {max(rtts):.2f} ms")
    print(f"  P95 RTT    : {sorted(rtts)[p95_idx]:.2f} ms")
    if server_lats:
        print(f"  Avg server : {statistics.mean(server_lats):.2f} ms  (inference only)")

    avg = statistics.mean(rtts)
    if avg < 100:
        print(f"  [PASS] Target <100ms tercapai ✓  (avg {avg:.1f}ms)")
    elif avg < 200:
        print(f"  [INFO] Latency {avg:.1f}ms — aman untuk interval 10-detik ESP32.")
    else:
        print(f"  [WARN] Rata-rata {avg:.1f}ms melebihi 200ms. Periksa koneksi jaringan.")
    return True


def test_invalid_input(base_url: str) -> bool:
    """Verify that the API returns correct error codes for bad inputs."""
    print(f"\n{'='*55}")
    print("TEST: error handling")
    print("="*55)
    all_pass = True

    # Case 1: wrong content type
    try:
        files = {"image": ("test.txt", b"this is not an image", "text/plain")}
        resp  = requests.post(f"{base_url}/predict", files=files, timeout=10)
        if resp.status_code == 415:
            print("  [PASS] text/plain → 415 Unsupported Media Type")
        else:
            print(f"  [FAIL] Expected 415, got {resp.status_code}: {resp.text[:200]}")
            all_pass = False
    except Exception as exc:
        print(f"  [FAIL] wrong content type: {exc}")
        all_pass = False

    # Case 2: empty image body
    try:
        files = {"image": ("empty.jpg", b"", "image/jpeg")}
        resp  = requests.post(f"{base_url}/predict", files=files, timeout=10)
        if resp.status_code == 400:
            print("  [PASS] empty file → 400 Bad Request")
        else:
            print(f"  [FAIL] Expected 400, got {resp.status_code}: {resp.text[:200]}")
            all_pass = False
    except Exception as exc:
        print(f"  [FAIL] empty body: {exc}")
        all_pass = False

    # Case 3: corrupt image bytes
    try:
        files = {"image": ("corrupt.jpg", b"\x00\x01\x02\x03garbage_data", "image/jpeg")}
        resp  = requests.post(f"{base_url}/predict", files=files, timeout=10)
        if resp.status_code in (422, 500):
            print(f"  [PASS] corrupt bytes → {resp.status_code} (422 or 500 expected)")
        else:
            print(f"  [FAIL] Expected 422/500, got {resp.status_code}: {resp.text[:200]}")
            all_pass = False
    except Exception as exc:
        print(f"  [FAIL] corrupt image: {exc}")
        all_pass = False

    # Case 4: file too large (simulate 6MB)
    try:
        big = b"\xff\xd8\xff" + b"\x00" * (6 * 1024 * 1024)   # fake JPEG header + 6MB
        files = {"image": ("big.jpg", big, "image/jpeg")}
        resp  = requests.post(f"{base_url}/predict", files=files, timeout=15)
        if resp.status_code in (413, 422, 400):
            print(f"  [PASS] oversized file → {resp.status_code}")
        else:
            print(f"  [INFO] Oversized file → {resp.status_code} (may be caught by Nginx)")
    except Exception as exc:
        print(f"  [INFO] oversized test: {exc}")

    return all_pass


# ----------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------

def main() -> None:
    """Parse arguments and run selected test suite."""
    parser = argparse.ArgumentParser(
        description="Test Clothing Classifier GCE endpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the GCE endpoint (e.g. http://34.x.x.x)",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to a test image (JPEG/PNG)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of iterations for latency test",
    )
    args = parser.parse_args()

    print(f"\nTarget URL : {args.url}")
    print(f"Image path : {args.image or '(not provided — predict/latency tests skipped)'}")

    results: dict[str, bool] = {}

    results["health"] = test_health(args.url)

    if args.image:
        results["predict"] = test_predict(args.url, args.image)
        results["latency"] = test_latency(args.url, args.image, n=args.runs)
    else:
        print("\n[INFO] --image not provided. Skipping predict and latency tests.")

    results["error_handling"] = test_invalid_input(args.url)

    # Final summary
    print(f"\n{'='*55}")
    print("SUMMARY")
    print("="*55)
    all_pass = True
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False
    print()

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
