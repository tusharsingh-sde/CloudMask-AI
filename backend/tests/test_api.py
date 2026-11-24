import pytest
import numpy as np
import api
from conftest import (
    build_4band_payload,
    make_16bit_tiff_bytes,
    make_png_bytes,
    make_oversized_png_bytes,
    make_too_small_png_bytes,
    make_corrupt_bytes,
)


def test_health_check_endpoint(client):
    """GET /api/health returns 200 with status online and model metadata."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert data["service"] == "CloudMask AI Backend"
    assert data["model_loaded"] is True
    assert "device" in data
    assert "checkpoint" in data


def test_readiness_probe_success_and_failure(client):
    """GET /api/ready returns 200 when model is loaded, 503 when unloaded."""
    res = client.get("/api/ready")
    assert res.status_code == 200
    assert res.json()["ready"] is True

    # Simulate model unready
    original_state = client.app.state.model_loaded
    client.app.state.model_loaded = False
    try:
        res_fail = client.get("/api/ready")
        assert res_fail.status_code == 503
        assert res_fail.json()["detail"]["ready"] is False
    finally:
        client.app.state.model_loaded = original_state


def test_predict_blocks_when_model_unloaded(client):
    """POST /api/predict immediately returns 503 if model weights are not loaded."""
    original_state = client.app.state.model_loaded
    client.app.state.model_loaded = False
    try:
        payload = build_4band_payload(64, 64)
        res = client.post("/api/predict", files=payload)
        assert res.status_code == 503
        assert "Model weights are not loaded" in res.json()["detail"]
    finally:
        client.app.state.model_loaded = original_state


def test_predict_valid_payload_and_schema(client):
    """POST /api/predict succeeds on valid 4-band PNGs with correct metrics and no imageUrl bloat."""
    payload = build_4band_payload(64, 64)
    res = client.post("/api/predict", files=payload, data={"threshold": "0.5", "overlay_opacity": "0.45"})
    assert res.status_code == 200

    data = res.json()
    assert data["status"] == "success"
    assert "imageUrl" not in data, "Redundant duplicate 'imageUrl' must be removed!"
    assert "rgb_composite_url" in data
    assert "mask_url" in data
    assert "overlay_url" in data
    assert "heatmap_url" in data

    # Quantitative coverage assertions
    cloud_pct = data["cloud_coverage_percentage"]
    clear_pct = data["clear_sky_percentage"]
    assert 0.0 <= cloud_pct <= 100.0
    assert 0.0 <= clear_pct <= 100.0
    assert round(cloud_pct + clear_pct, 1) == 100.0

    total_px = data["total_pixels"]
    assert total_px == 64 * 64
    assert data["cloud_pixels"] + data["clear_pixels"] == total_px

    meta = data["metadata"]
    assert meta["dimensions"] == "64 x 64 px"
    assert meta["threshold"] == 0.5
    assert meta["overlay_opacity"] == 0.45


def test_radiometric_scaling_16bit_preservation():
    """Verify decode_band_image linearly scales dark uint16 (500-1500) rather than stretching to 255."""
    dark_16bit_bytes = make_16bit_tiff_bytes(width=64, height=64, min_val=500, max_val=1500)
    decoded = api.decode_band_image(dark_16bit_bytes, "dark_scene.tif")

    assert decoded.dtype == np.uint8
    assert decoded.shape == (64, 64)
    # 1500 / 65535 * 255 ≈ 5.8 -> uint8 max <= 10
    assert decoded.max() <= 10, f"Dark clear-sky 16-bit scene was stretched to {decoded.max()}!"


def test_dimension_harmonization(client):
    """Mismatched band resolutions harmonize to the red band's target dimension."""
    red_bytes = make_png_bytes(128, 128)
    other_bytes = make_png_bytes(64, 64)

    payload = {
        "red": ("red.png", red_bytes, "image/png"),
        "green": ("green.png", other_bytes, "image/png"),
        "blue": ("blue.png", other_bytes, "image/png"),
        "nir": ("nir.png", other_bytes, "image/png"),
    }

    res = client.post("/api/predict", files=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["metadata"]["dimensions"] == "128 x 128 px"
    assert data["total_pixels"] == 128 * 128


def test_tiled_inference_large_scene(client):
    """512x512 scene triggers tiled_inference=True and returns non-NaN valid output."""
    payload = build_4band_payload(512, 512)
    res = client.post("/api/predict", files=payload)
    assert res.status_code == 200

    data = res.json()
    assert data["metadata"]["tiled_inference"] is True
    assert data["metadata"]["tile_size"] == 384
    assert data["metadata"]["dimensions"] == "512 x 512 px"
    assert not np.isnan(data["cloud_coverage_percentage"])


def test_security_invalid_extension(client):
    """Band with disallowed extension returns HTTP 400."""
    payload = build_4band_payload(64, 64)
    payload["red"] = ("red.txt", b"plain text", "text/plain")
    res = client.post("/api/predict", files=payload)
    assert res.status_code == 400
    assert "invalid extension" in res.json()["detail"].lower()


def test_security_empty_file(client):
    """Empty band upload returns HTTP 400."""
    payload = build_4band_payload(64, 64)
    payload["nir"] = ("nir.png", b"", "image/png")
    res = client.post("/api/predict", files=payload)
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()


def test_security_oversized_and_too_small_headers(client):
    """Pre-decode decompression bomb guard rejects headers > 4096px or < 16px with HTTP 400."""
    # Oversized 5000x200
    payload_huge = build_4band_payload(64, 64)
    payload_huge["red"] = ("huge.png", make_oversized_png_bytes(), "image/png")
    res_huge = client.post("/api/predict", files=payload_huge)
    assert res_huge.status_code == 400
    assert "exceeds maximum allowed" in res_huge.json()["detail"]

    # Too small 8x8
    payload_small = build_4band_payload(64, 64)
    payload_small["red"] = ("small.png", make_too_small_png_bytes(), "image/png")
    res_small = client.post("/api/predict", files=payload_small)
    assert res_small.status_code == 400
    assert "too small" in res_small.json()["detail"]


def test_validation_out_of_bounds_parameters(client):
    """Out-of-bounds threshold or overlay_opacity returns HTTP 422 Unprocessable Entity."""
    payload = build_4band_payload(64, 64)

    res_thresh = client.post("/api/predict", files=payload, data={"threshold": "5.0"})
    assert res_thresh.status_code == 422

    res_opacity = client.post("/api/predict", files=payload, data={"overlay_opacity": "0.0"})
    assert res_opacity.status_code == 422


def test_security_headers_and_cors(client):
    """Verify security headers (nosniff, DENY, no-store) and CORS configuration."""
    payload = build_4band_payload(64, 64)
    res = client.post("/api/predict", files=payload)
    assert res.status_code == 200

    headers = res.headers
    assert headers.get("X-Content-Type-Options") == "nosniff"
    assert headers.get("X-Frame-Options") == "DENY"
    assert headers.get("Referrer-Policy") == "no-referrer"
    assert "no-store" in headers.get("Cache-Control", "")

    # CORS preflight check
    cors_res = client.options(
        "/api/predict",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        }
    )
    assert cors_res.status_code == 200
    assert cors_res.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert cors_res.headers.get("access-control-allow-credentials") is None or cors_res.headers.get("access-control-allow-credentials") == "false"


def test_sliding_window_rate_limiter(client, monkeypatch):
    """Bursting requests beyond RATE_LIMIT_REQUESTS returns HTTP 429 with Retry-After header."""
    monkeypatch.setattr(api, "RATE_LIMIT_ENABLED", True)
    api.rate_limiter.reset()
    api.rate_limiter.max_requests = 3
    api.rate_limiter.window_seconds = 10

    payload = build_4band_payload(64, 64)

    # 3 allowed requests
    for _ in range(3):
        r = client.post("/api/predict", files=payload)
        assert r.status_code == 200

    # 4th request rate-limited
    r_blocked = client.post("/api/predict", files=payload)
    assert r_blocked.status_code == 429
    assert "Retry-After" in r_blocked.headers
    assert "Rate limit exceeded" in r_blocked.json()["detail"]

    # Reset
    api.rate_limiter.reset()
