/**
 * CloudMask AI API Service Client
 * Handles robust network calls, timeout control, input packaging, and sanitization.
 */

const BASE_URL = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

/**
 * Checks backend connectivity and health status.
 */
export async function checkBackendHealth() {
  const url = BASE_URL ? `${BASE_URL}/api/health` : "/api/health";
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 5000);

  try {
    const response = await fetch(url, {
      method: "GET",
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`Health check failed with HTTP ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    clearTimeout(timeoutId);
    return {
      status: "offline",
      error: error.name === "AbortError" ? "Connection timed out" : error.message,
    };
  }
}

/**
 * Submits the 4 satellite bands and analysis settings to the deep learning backend.
 * @param {Object} files Object mapping { red, green, blue, nir } to File objects
 * @param {Object} options Optional settings: { threshold, overlayOpacity }
 */
export async function predictCloudMask(files, options = {}) {
  const { threshold = 0.5, overlayOpacity = 0.45 } = options;

  // Validate that all required bands exist
  const requiredBands = ["red", "green", "blue", "nir"];
  const missing = requiredBands.filter((band) => !files[band]);

  if (missing.length > 0) {
    throw new Error(`Missing required bands: ${missing.join(", ")}`);
  }

  const formData = new FormData();
  for (const band of requiredBands) {
    formData.append(band, files[band]);
  }
  formData.append("threshold", String(threshold));
  formData.append("overlay_opacity", String(overlayOpacity));

  // Determine endpoint
  const url = BASE_URL ? `${BASE_URL}/api/predict` : "/api/predict";

  const controller = new AbortController();
  // 60-second timeout for large satellite image processing
  const timeoutId = setTimeout(() => controller.abort(), 60000);

  try {
    const response = await fetch(url, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      let errorMessage = `Server error (HTTP ${response.status})`;
      try {
        const errorData = await response.json();
        if (errorData.detail) {
          errorMessage = typeof errorData.detail === "string" 
            ? errorData.detail 
            : JSON.stringify(errorData.detail);
        } else if (errorData.error) {
          errorMessage = errorData.error;
        }
      } catch {
        // Fallback to text status
      }
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === "AbortError") {
      throw new Error("Request timed out. The server took longer than 60 seconds to process the images.");
    }
    throw err;
  }
}
