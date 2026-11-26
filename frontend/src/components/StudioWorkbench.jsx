import { useState, useRef, useEffect, useCallback } from "react";
import {
  UploadCloud,
  X,
  RotateCcw,
  Sliders,
  AlertCircle,
  Layers,
  Download,
  Maximize2,
  Split,
  Eye,
  Grid2X2,
  CheckCircle2,
  Sparkles,
  ArrowRight,
} from "lucide-react";
import { BANDS } from "../constants/bands";
import { SAMPLE_SCENES } from "../constants/samples";

const isTiffFile = (file) => {
  if (!file) return false;
  return (
    /\.(tif|tiff)$/i.test(file.name || "") ||
    file.type === "image/tiff" ||
    file.type === "image/x-tiff"
  );
};

export default function StudioWorkbench({
  files,
  onFilesChange,
  onResetFiles,
  onSubmit,
  loading,
  error,
  threshold,
  setThreshold,
  opacity,
  setOpacity,
  result,
  runHistory = [],
  onSelectRun,
  backendHealth,
  activeSampleId = null,
  loadingSampleId = null,
  onLoadSample,
  onClearActiveSample,
}) {
  // Input UI State
  const [multiDropActive, setMultiDropActive] = useState(false);
  const [dragOverBand, setDragOverBand] = useState(null);
  const [autoMatchToast, setAutoMatchToast] = useState(null);
  const fileInputRefs = useRef({});

  // Viewport UI State
  const [viewerMode, setViewerMode] = useState("split"); // "split" | "single" | "quad"
  const [splitRightChannel, setSplitRightChannel] = useState("overlay"); // "overlay" | "mask" | "heatmap"
  const [singleChannel, setSingleChannel] = useState("overlay");
  const [splitPosition, setSplitPosition] = useState(50); // percentage 0..100
  const [isDraggingSplit, setIsDraggingSplit] = useState(false);
  const [fullscreenImage, setFullscreenImage] = useState(null);
  const [previewUrls, setPreviewUrls] = useState({});
  const [imageDecodeErrors, setImageDecodeErrors] = useState({});

  const splitStageRef = useRef(null);

  // Manage Object URLs safely for uploaded image previews (PNG/JPG/WEBP only, skipping GeoTIFF)
  useEffect(() => {
    let active = true;
    const urls = {};
    BANDS.forEach((band) => {
      const file = files[band.id];
      if (
        file &&
        !isTiffFile(file) &&
        (file.type.startsWith("image/") || /\.(png|jpe?g|webp)$/i.test(file.name || ""))
      ) {
        try {
          urls[band.id] = URL.createObjectURL(file);
        } catch {
          // ignore
        }
      }
    });
    queueMicrotask(() => {
      if (active) setPreviewUrls(urls);
    });

    return () => {
      active = false;
      Object.values(urls).forEach((u) => {
        try {
          URL.revokeObjectURL(u);
        } catch {
          // ignore
        }
      });
    };
  }, [files]);

  // Escape key listener for fullscreen modal
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === "Escape") setFullscreenImage(null);
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // Multi-band Auto Matching Logic
  const handleBatchFiles = useCallback((fileList) => {
    if (!fileList || fileList.length === 0) return;

    const updated = { ...files };
    const unassigned = [];
    let matchedCount = 0;

    Array.from(fileList).forEach((file) => {
      const name = file.name.toLowerCase();
      let matched = null;

      if (name.includes("red") || name.includes("b04") || name.includes("b4")) {
        matched = "red";
      } else if (name.includes("green") || name.includes("b03") || name.includes("b3")) {
        matched = "green";
      } else if (name.includes("blue") || name.includes("b02") || name.includes("b2")) {
        matched = "blue";
      } else if (
        name.includes("nir") ||
        name.includes("b05") ||
        name.includes("b5") ||
        name.includes("b08") ||
        name.includes("b8")
      ) {
        matched = "nir";
      }

      if (matched) {
        updated[matched] = file;
        matchedCount++;
      } else {
        unassigned.push(file);
      }
    });

    // Fill remaining empty slots with unassigned files in standard order
    const order = ["red", "green", "blue", "nir"];
    const emptySlots = order.filter((b) => !updated[b]);

    unassigned.forEach((file, i) => {
      if (i < emptySlots.length) {
        updated[emptySlots[i]] = file;
        matchedCount++;
      }
    });

    onClearActiveSample?.();
    onFilesChange(updated);
    setAutoMatchToast(`${matchedCount} spectral band${matchedCount > 1 ? "s" : ""} assigned.`);
    setTimeout(() => setAutoMatchToast(null), 3500);
  }, [files, onFilesChange, onClearActiveSample]);

  // Individual Band Upload / Clear
  const handleSingleFile = (bandId, file) => {
    onClearActiveSample?.();
    onFilesChange({
      ...files,
      [bandId]: file,
    });
  };

  const handleClearBand = (bandId, e) => {
    e.stopPropagation();
    onClearActiveSample?.();
    const updated = { ...files };
    delete updated[bandId];
    if (fileInputRefs.current[bandId]) {
      fileInputRefs.current[bandId].value = "";
    }
    onFilesChange(updated);
  };

  const handleResetAll = () => {
    onClearActiveSample?.();
    Object.keys(fileInputRefs.current).forEach((k) => {
      if (fileInputRefs.current[k]) fileInputRefs.current[k].value = "";
    });
    onResetFiles();
  };

  // Split Slider Pointer Movement
  const updateSplitPosFromEvent = useCallback((clientX) => {
    if (!splitStageRef.current) return;
    const rect = splitStageRef.current.getBoundingClientRect();
    const pos = ((clientX - rect.left) / rect.width) * 100;
    setSplitPosition(Math.max(2, Math.min(98, pos)));
  }, []);

  const handleSplitPointerDown = (e) => {
    setIsDraggingSplit(true);
    updateSplitPosFromEvent(e.clientX || (e.touches && e.touches[0].clientX));
  };

  useEffect(() => {
    if (!isDraggingSplit) return;

    const handlePointerMove = (e) => {
      const clientX = e.clientX !== undefined ? e.clientX : e.touches?.[0]?.clientX;
      if (clientX !== undefined) updateSplitPosFromEvent(clientX);
    };

    const handlePointerUp = () => {
      setIsDraggingSplit(false);
    };

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
    window.addEventListener("touchmove", handlePointerMove);
    window.addEventListener("touchend", handlePointerUp);

    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
      window.removeEventListener("touchmove", handlePointerMove);
      window.removeEventListener("touchend", handlePointerUp);
    };
  }, [isDraggingSplit, updateSplitPosFromEvent]);

  // Image Download Helpers
  const downloadSingleImage = (url, name) => {
    if (!url) return;
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const downloadAllImages = () => {
    if (!result) return;
    const items = [
      { url: result.rgb_composite_url, name: "cloudmask_rgb_truecolor.png" },
      { url: result.overlay_url, name: "cloudmask_overlay.png" },
      { url: result.mask_url, name: "cloudmask_binary_mask.png" },
      { url: result.heatmap_url, name: "cloudmask_probability_heatmap.png" },
    ];
    items.forEach((item, idx) => {
      setTimeout(() => downloadSingleImage(item.url, item.name), idx * 250);
    });
  };

  const loadedCount = BANDS.filter((b) => files[b.id]).length;
  const allLoaded = loadedCount === 4;

  const channelMap = {
    overlay: { name: "Mask Overlay", url: result?.overlay_url, badge: `OVERLAY (${Math.round(opacity * 100)}%)` },
    mask: { name: "Binary Mask", url: result?.mask_url, badge: `BINARY MASK (${Math.round(threshold * 100)}%)` },
    heatmap: { name: "Probability Heatmap", url: result?.heatmap_url, badge: "MAGMA HEATMAP" },
    rgb: { name: "RGB Composite", url: result?.rgb_composite_url, badge: "TRUE COLOR (B4·B3·B2)" },
  };

  return (
    <div id="studio" className="workbench-section">
      <div className="workbench-container">
        {/* Spectral Input & Inference Controls */}
        <aside className="workbench-sidebar">
          <div className="sidebar-header">
            <div className="header-eyebrow">
              <span className="accent-square" />
              <span>SATELLITE DATA INGESTION</span>
            </div>
            <h2>Spectral Workbench</h2>
            <p>Load 4 synchronized bands for convolutional cloud mask segmentation.</p>
          </div>

          {/* Smart Multi-Band Dropzone */}
          <div
            className={`smart-dropzone ${multiDropActive ? "dropzone-active" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setMultiDropActive(true);
            }}
            onDragLeave={(e) => {
              e.preventDefault();
              setMultiDropActive(false);
            }}
            onDrop={(e) => {
              e.preventDefault();
              setMultiDropActive(false);
              handleBatchFiles(e.dataTransfer.files);
            }}
            onClick={() => fileInputRefs.current.multi?.click()}
          >
            <div className="dropzone-icon">
              <UploadCloud size={20} className="icon-cyan" />
            </div>
            <div className="dropzone-text">
              <span className="dropzone-title">Drop all 4 spectral bands at once</span>
              <span className="dropzone-sub">Auto-matched by filename (R, G, B, NIR) or browse</span>
            </div>
            <input
              ref={(el) => (fileInputRefs.current.multi = el)}
              type="file"
              multiple
              style={{ display: "none" }}
              accept=".tif,.tiff,.png,.jpg,.jpeg"
              onChange={(e) => handleBatchFiles(e.target.files)}
            />
          </div>

          {/* Quick-Load Sample Scenes */}
          <div className="sample-scenes-block">
            <div className="sample-scenes-header">
              <div className="sample-scenes-eyebrow">
                <Sparkles size={12} className="icon-cyan" />
                <span>QUICK-LOAD SAMPLE SCENES</span>
              </div>
              <span className="sample-scenes-badge">1-CLICK TEST</span>
            </div>

            <div className="sample-scenes-list">
              {SAMPLE_SCENES.map((scene) => {
                const isActive = activeSampleId === scene.id && allLoaded;
                const isFetching = loadingSampleId === scene.id;
                const baseUrl = (import.meta.env.BASE_URL || "/").replace(/\/$/, "");
                const previewSrc = `${baseUrl}/samples/${scene.folder}/preview.png`;

                return (
                  <button
                    key={scene.id}
                    type="button"
                    className={`sample-scene-card ${isActive ? "active-sample" : ""}`}
                    onClick={() => onLoadSample?.(scene)}
                    disabled={loading || loadingSampleId !== null}
                    title={`1-Click Test: ${scene.title} (${scene.tag})`}
                  >
                    <div className="sample-thumb-wrap">
                      <img
                        src={previewSrc}
                        alt={scene.title}
                        className="sample-thumb-img"
                        loading="lazy"
                      />
                    </div>

                    <div className="sample-scene-info">
                      <div className="sample-scene-title-row">
                        <span className="sample-scene-name">{scene.title}</span>
                        {isActive && <span className="sample-active-dot" title="Active Scene" />}
                      </div>
                      <div className="sample-scene-meta-row">
                        {isFetching ? (
                          <span className="sample-loading-status">
                            <span className="sample-mini-spinner" />
                            <span>FETCHING...</span>
                          </span>
                        ) : (
                          <span className="sample-scene-tag">{scene.tag}</span>
                        )}
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {autoMatchToast && (
            <div className="match-toast">
              <Sparkles size={14} className="toast-icon" />
              <span>{autoMatchToast}</span>
            </div>
          )}

          {/* 4 Spectral Band Slots (2x2 Grid) */}
          <div className="band-slots-grid">
            {BANDS.map((band) => {
              const file = files[band.id];
              const preview = previewUrls[band.id];
              const isTiff = isTiffFile(file);
              const fileKey = file ? `${file.name}_${file.size}_${file.lastModified}` : "";
              const hasDecodeError = Boolean(file && imageDecodeErrors[band.id] === fileKey);
              const hasPreview = Boolean(file && preview && !hasDecodeError && !isTiff);
              const isOver = dragOverBand === band.id;

              return (
                <div
                  key={band.id}
                  className={`band-slot-card ${file ? "has-file" : ""} ${isOver ? "slot-over" : ""}`}
                  style={{ "--band-accent": band.color }}
                  onDragOver={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    setDragOverBand(band.id);
                  }}
                  onDragLeave={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    if (dragOverBand === band.id) setDragOverBand(null);
                  }}
                  onDrop={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    setDragOverBand(null);
                    const dropped = e.dataTransfer.files?.[0];
                    if (dropped) handleSingleFile(band.id, dropped);
                  }}
                  onClick={() => fileInputRefs.current[band.id]?.click()}
                >
                  <div className="card-accent-bar" />

                  <div className="card-top-row">
                    <span className="band-badge">{band.badge}</span>
                    <span className="band-name-code">{band.name}</span>
                    {file ? (
                      <button
                        type="button"
                        className="clear-band-btn"
                        onClick={(e) => handleClearBand(band.id, e)}
                        title={`Remove ${band.name}`}
                      >
                        <X size={12} />
                      </button>
                    ) : (
                      <span className="slot-empty-dot" />
                    )}
                  </div>

                  <div className="card-body">
                    {hasPreview ? (
                      <div className="slot-thumbnail-wrap">
                        <img
                          src={preview}
                          alt={band.name}
                          className="slot-thumbnail"
                          onError={() => {
                            if (file) {
                              setImageDecodeErrors((prev) => ({
                                ...prev,
                                [band.id]: `${file.name}_${file.size}_${file.lastModified}`,
                              }));
                            }
                          }}
                        />
                      </div>
                    ) : file ? (
                      <div className="geotiff-preview-state">
                        <div
                          className="geotiff-icon-box"
                          style={{
                            backgroundColor: `${band.color}15`,
                            borderColor: `${band.color}35`,
                          }}
                        >
                          <Layers size={18} className="geotiff-icon" style={{ color: band.color }} />
                        </div>
                        <div className="geotiff-details">
                          <span className="geotiff-badge">16-BIT GEOTIFF</span>
                          <span className="geotiff-status">
                            <span className="geotiff-status-dot" />
                            <span>READY · RAW BAND</span>
                          </span>
                        </div>
                      </div>
                    ) : (
                      <div className="slot-placeholder">
                        <span className="slot-letter">{band.letter}</span>
                        <span className="slot-wavelength">{band.description.split("(")[0]}</span>
                      </div>
                    )}
                  </div>

                  <div className="card-footer">
                    {file ? (
                      <div className="file-meta">
                        <span className="file-name" title={file.name}>
                          {file.name}
                        </span>
                        <span className="file-size">
                          {(file.size / 1024).toFixed(0)} KB
                        </span>
                      </div>
                    ) : (
                      <span className="slot-action-hint">Click or drop {band.letter}</span>
                    )}
                  </div>

                  <input
                    ref={(el) => (fileInputRefs.current[band.id] = el)}
                    type="file"
                    style={{ display: "none" }}
                    accept=".tif,.tiff,.png,.jpg,.jpeg"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) handleSingleFile(band.id, f);
                    }}
                  />
                </div>
              );
            })}
          </div>

          {/* Inference Parameters */}
          <div className="parameters-card">
            <div className="param-header">
              <Sliders size={14} className="icon-cyan" />
              <span>Inference Hyperparameters</span>
            </div>

            <div className="param-item">
              <div className="param-label-row">
                <span className="param-name">Confidence Threshold</span>
                <span className="param-val-pill">{Math.round(threshold * 100)}%</span>
              </div>
              <input
                type="range"
                min="0.10"
                max="0.90"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="param-slider"
              />
              <span className="param-hint">Probability cutoff for cloud classification</span>
            </div>

            <div className="param-item">
              <div className="param-label-row">
                <span className="param-name">Overlay Opacity</span>
                <span className="param-val-pill">{Math.round(opacity * 100)}%</span>
              </div>
              <input
                type="range"
                min="0.10"
                max="1.00"
                step="0.05"
                value={opacity}
                onChange={(e) => setOpacity(parseFloat(e.target.value))}
                className="param-slider"
              />
              <span className="param-hint">Alpha transparency of cyan cloud mask over RGB</span>
            </div>
          </div>

          {/* Error Banner */}
          {error && (
            <div className="error-alert">
              <AlertCircle size={16} className="error-icon" />
              <div className="error-content">
                <strong>Inference Error</strong>
                <span>{error}</span>
              </div>
            </div>
          )}

          {/* Action CTAs */}
          <div className="action-button-row">
            <button
              type="button"
              className="reset-btn"
              onClick={handleResetAll}
              disabled={loading || loadedCount === 0}
            >
              <RotateCcw size={14} />
              <span>Reset</span>
            </button>

            <button
              type="button"
              className="run-inference-btn"
              onClick={onSubmit}
              disabled={loading || !allLoaded}
            >
              {loading ? (
                <>
                  <span className="btn-spinner" />
                  <span>Running U-Net Inference...</span>
                </>
              ) : (
                <>
                  <span>Run Cloud Segmentation</span>
                  <ArrowRight size={16} />
                </>
              )}
            </button>
          </div>
        </aside>

        {/* Live Inspection Viewport */}
        <main className="workbench-stage">
          <div className="stage-topbar">
            <div className="stage-title-wrap">
              <div className="stage-title-eyebrow">
                <span className="pulse-cyan-dot" />
                <span>MULTISPECTRAL INSPECTION VIEWPORT</span>
              </div>
              <h3>Radiometric Telemetry & Analysis</h3>
            </div>

            {runHistory.length > 1 && (
              <div className="history-pill-strip">
                <span className="history-label">Runs:</span>
                {runHistory.map((item, idx) => (
                  <button
                    key={item.id || idx}
                    className={`history-chip ${result?.id === item.id ? "active-run" : ""}`}
                    onClick={() => onSelectRun(item)}
                  >
                    #{idx + 1} ({Math.round(item.metadata?.threshold * 100)}% → {item.cloud_coverage_percentage}%)
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Loading State */}
          {loading && (
            <div className="stage-panel stage-loading">
              <div className="radar-sweep-container">
                <div className="radar-grid" />
                <div className="radar-sweep-beam" />
                <div className="radar-reticle" />
              </div>
              <div className="loading-caption">
                <span className="loading-headline">Executing PyTorch U-Net Inference</span>
                <span className="loading-sub">
                  Harmonizing 4 spectral tensors · Feed-forward segmentation on {backendHealth?.device?.toUpperCase() || "CPU"}
                </span>
              </div>
            </div>
          )}

          {/* Standby State */}
          {!loading && !result && (
            <div className="stage-panel stage-standby">
              <div className="standby-crosshair top-left" />
              <div className="standby-crosshair top-right" />
              <div className="standby-crosshair bottom-left" />
              <div className="standby-crosshair bottom-right" />

              <div className="standby-briefing-card">
                <div className="briefing-radar-icon">
                  <Layers size={32} className="icon-cyan" />
                </div>
                <h4>Tactical Mission Briefing</h4>
                <p>
                  Ground station engine in standby. Upload your 4 satellite bands (R, G, B, NIR) on the left to initialize the in-memory U-Net segmentation pipeline.
                </p>

                {/* 3-Step Pipeline Flow */}
                <div className="pipeline-steps-row">
                  <div className={`pipe-step ${loadedCount >= 1 ? "done" : ""}`}>
                    <span className="step-num">01</span>
                    <span className="step-title">Load B02–B05</span>
                    <span className="step-badge">{loadedCount}/4 Ready</span>
                  </div>
                  <div className="pipe-arrow">→</div>
                  <div className={`pipe-step ${allLoaded ? "ready" : ""}`}>
                    <span className="step-num">02</span>
                    <span className="step-title">U-Net Forward</span>
                    <span className="step-badge">{backendHealth?.device?.toUpperCase() || "CPU"} Engine</span>
                  </div>
                  <div className="pipe-arrow">→</div>
                  <div className="pipe-step">
                    <span className="step-num">03</span>
                    <span className="step-title">Curtain Compare</span>
                    <span className="step-badge">4 Channels</span>
                  </div>
                </div>

                <div className="standby-footer-status">
                  <span className={`status-dot ${backendHealth?.status === "online" ? "dot-online" : "dot-offline"}`} />
                  <span>
                    {backendHealth?.status === "online"
                      ? `Neural Inference Core Online · ${backendHealth?.device?.toUpperCase() || "CPU"}`
                      : "Awaiting local inference server at /api/health"}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Results View */}
          {!loading && result && (
            <div className="stage-panel stage-results">
              <div className="kpi-telemetry-strip">
                <div className="kpi-card kpi-cloud">
                  <div className="kpi-num">{result.cloud_coverage_percentage}%</div>
                  <div className="kpi-label">Cloud Coverage</div>
                  <div className="kpi-bar-track">
                    <div
                      className="kpi-bar-fill cloud-fill"
                      style={{ width: `${Math.min(100, result.cloud_coverage_percentage)}%` }}
                    />
                  </div>
                  <span className="kpi-sub">
                    {result.cloud_pixels?.toLocaleString()} px
                  </span>
                </div>

                <div className="kpi-card kpi-clear">
                  <div className="kpi-num">{result.clear_sky_percentage}%</div>
                  <div className="kpi-label">Clear Sky / Ground</div>
                  <div className="kpi-bar-track">
                    <div
                      className="kpi-bar-fill clear-fill"
                      style={{ width: `${Math.min(100, result.clear_sky_percentage)}%` }}
                    />
                  </div>
                  <span className="kpi-sub">
                    {((result.total_pixels || 0) - (result.cloud_pixels || 0)).toLocaleString()} px
                  </span>
                </div>

                <div className="kpi-card">
                  <div className="kpi-num">
                    {result.total_pixels ? result.total_pixels.toLocaleString() : "147,456"}
                  </div>
                  <div className="kpi-label">Total Pixels Analyzed</div>
                  <span className="kpi-sub">{result.metadata?.dimensions || "384 × 384 Array"}</span>
                </div>

                <div className="kpi-card">
                  <div className="kpi-num">
                    {result.metadata?.inference_time_ms ? `${result.metadata.inference_time_ms} ms` : "< 1s"}
                  </div>
                  <div className="kpi-label">Inference Latency</div>
                  <span className="kpi-sub">
                    Device: {result.metadata?.device?.toUpperCase() || "CPU"}
                  </span>
                </div>
              </div>

              <div className="viewport-toolbar">
                <div className="mode-switcher-group">
                  <button
                    className={`mode-tab-btn ${viewerMode === "split" ? "active" : ""}`}
                    onClick={() => setViewerMode("split")}
                  >
                    <Split size={14} />
                    <span>Split Compare</span>
                  </button>
                  <button
                    className={`mode-tab-btn ${viewerMode === "single" ? "active" : ""}`}
                    onClick={() => setViewerMode("single")}
                  >
                    <Eye size={14} />
                    <span>Single View</span>
                  </button>
                  <button
                    className={`mode-tab-btn ${viewerMode === "quad" ? "active" : ""}`}
                    onClick={() => setViewerMode("quad")}
                  >
                    <Grid2X2 size={14} />
                    <span>Quad Grid</span>
                  </button>
                </div>

                {/* Sub-selector when in Split Mode */}
                {viewerMode === "split" && (
                  <div className="sub-layer-pills">
                    <span className="layer-pill-label">Right Layer:</span>
                    {["overlay", "mask", "heatmap"].map((id) => (
                      <button
                        key={id}
                        className={`sub-pill ${splitRightChannel === id ? "active-pill" : ""}`}
                        onClick={() => setSplitRightChannel(id)}
                      >
                        {channelMap[id].name}
                      </button>
                    ))}
                  </div>
                )}

                {/* Sub-selector when in Single View Mode */}
                {viewerMode === "single" && (
                  <div className="sub-layer-pills">
                    {["overlay", "mask", "rgb", "heatmap"].map((id) => (
                      <button
                        key={id}
                        className={`sub-pill ${singleChannel === id ? "active-pill" : ""}`}
                        onClick={() => setSingleChannel(id)}
                      >
                        {channelMap[id].name}
                      </button>
                    ))}
                  </div>
                )}

                {/* Export & Lightbox Actions */}
                <div className="toolbar-actions">
                  <button
                    className="tool-btn"
                    title="Fullscreen Lightbox"
                    onClick={() => {
                      const img =
                        viewerMode === "single"
                          ? channelMap[singleChannel]?.url
                          : viewerMode === "split"
                          ? channelMap[splitRightChannel]?.url
                          : result.overlay_url;
                      setFullscreenImage(img);
                    }}
                  >
                    <Maximize2 size={14} />
                  </button>
                  <button
                    className="tool-btn"
                    title="Export Current View"
                    onClick={() => {
                      const cur =
                        viewerMode === "single" ? singleChannel : splitRightChannel;
                      downloadSingleImage(channelMap[cur]?.url, `cloudmask_${cur}.png`);
                    }}
                  >
                    <Download size={14} />
                    <span>Export</span>
                  </button>
                  <button
                    className="tool-btn-accent"
                    title="Download all 4 channels"
                    onClick={downloadAllImages}
                  >
                    <Download size={14} />
                    <span>All 4 PNGs</span>
                  </button>
                </div>
              </div>

              <div className="inspection-stage-canvas">
                {viewerMode === "split" && (
                  <div
                    ref={splitStageRef}
                    className="split-comparison-container"
                    onPointerDown={handleSplitPointerDown}
                    tabIndex={0}
                    role="slider"
                    aria-label="Split Compare Slider"
                    aria-valuenow={Math.round(splitPosition)}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    onKeyDown={(e) => {
                      if (e.key === "ArrowLeft") setSplitPosition((p) => Math.max(0, p - 4));
                      if (e.key === "ArrowRight") setSplitPosition((p) => Math.min(100, p + 4));
                    }}
                  >
                    <img
                      src={result.rgb_composite_url}
                      alt="True Color RGB"
                      className="split-img left-img"
                    />

                    <div
                      className="split-right-wrapper"
                      style={{ clipPath: `inset(0 0 0 ${splitPosition}%)` }}
                    >
                      <img
                        src={channelMap[splitRightChannel]?.url}
                        alt={channelMap[splitRightChannel]?.name}
                        className="split-img right-img"
                      />
                    </div>

                    <div
                      className="split-divider-line"
                      style={{ left: `${splitPosition}%` }}
                    >
                      <div className="split-drag-handle">
                        <span className="handle-chevron left">‹</span>
                        <span className="handle-chevron right">›</span>
                      </div>
                    </div>

                    <div className="badge-corner top-left">
                      <span className="corner-tag tag-rgb">TRUE COLOR RGB (B4·B3·B2)</span>
                    </div>
                    <div className="badge-corner top-right">
                      <span className="corner-tag tag-layer">
                        {channelMap[splitRightChannel]?.badge}
                      </span>
                    </div>

                    <div className="split-instruction-tag">
                      <span>Drag curtain or press ← / →</span>
                    </div>
                  </div>
                )}

                {viewerMode === "single" && (
                  <div className="single-view-container">
                    <img
                      src={channelMap[singleChannel]?.url}
                      alt={channelMap[singleChannel]?.name}
                      className="single-main-image"
                      onClick={() => setFullscreenImage(channelMap[singleChannel]?.url)}
                    />
                    <div className="badge-corner top-left">
                      <span className="corner-tag tag-layer">
                        {channelMap[singleChannel]?.name}
                      </span>
                    </div>
                  </div>
                )}

                {viewerMode === "quad" && (
                  <div className="quad-matrix-grid">
                    {["rgb", "overlay", "mask", "heatmap"].map((id) => (
                      <div
                        key={id}
                        className="quad-tile"
                        onClick={() => setFullscreenImage(channelMap[id]?.url)}
                      >
                        <img
                          src={channelMap[id]?.url}
                          alt={channelMap[id]?.name}
                          className="quad-img"
                        />
                        <div className="quad-tag-bar">
                          <span className="quad-name">{channelMap[id]?.name}</span>
                          <Maximize2 size={12} className="tile-zoom-icon" />
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Telemetry Footer */}
              <div className="telemetry-footer-strip">
                <div className="telemetry-metric">
                  <span className="t-key">CONFIDENCE CUTOFF:</span>
                  <span className="t-val">
                    {result.metadata?.threshold != null
                      ? `${Math.round(result.metadata.threshold * 100)}%`
                      : "50%"}
                  </span>
                </div>
                <div className="telemetry-metric">
                  <span className="t-key">OVERLAY OPACITY:</span>
                  <span className="t-val">
                    {result.metadata?.overlay_opacity != null
                      ? `${Math.round(result.metadata.overlay_opacity * 100)}%`
                      : "45%"}
                  </span>
                </div>
                <div className="telemetry-metric">
                  <span className="t-key">MODEL EPOCH:</span>
                  <span className="t-val">{result.metadata?.model_epoch ?? "Best Checkpoint"}</span>
                </div>
                <div className="telemetry-metric">
                  <span className="t-key">VALIDATION LOSS:</span>
                  <span className="t-val">
                    {typeof result.metadata?.model_best_loss === "number"
                      ? result.metadata.model_best_loss.toFixed(4)
                      : "0.0958"}
                  </span>
                </div>
                <div className="telemetry-metric">
                  <span className="t-key">PIPELINE VERDICT:</span>
                  <span className="t-val text-emerald">
                    <CheckCircle2 size={12} /> Passed In-Memory
                  </span>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>

      {/* Fullscreen Lightbox Modal */}
      {fullscreenImage && (
        <div className="workbench-lightbox" onClick={() => setFullscreenImage(null)}>
          <div className="lightbox-inner" onClick={(e) => e.stopPropagation()}>
            <img src={fullscreenImage} alt="Enlarged Viewport" className="lightbox-img" />
            <button
              className="lightbox-close-btn"
              onClick={() => setFullscreenImage(null)}
              title="Close (Escape)"
            >
              <X size={18} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
