import { useState, useRef } from "react";
import { BANDS } from "../constants/bands";
import { UploadCloud, CheckCircle2, RotateCcw, ArrowRight, Sliders, AlertTriangle } from "lucide-react";

export default function ProcessingDashboard({
  files,
  onSelectFile,
  onResetFiles,
  onSubmit,
  loading,
  error,
  threshold,
  setThreshold,
  opacity,
  setOpacity
}) {
  const [dragOverBand, setDragOverBand] = useState(null);
  const fileInputRefs = useRef({});

  const allFilesUploaded = BANDS.every((band) => files[band.id]);
  const uploadedCount = BANDS.filter((band) => files[band.id]).length;

  const handleDragOver = (e, bandId) => {
    e.preventDefault();
    setDragOverBand(bandId);
  };

  const handleDragLeave = (e, bandId) => {
    e.preventDefault();
    if (dragOverBand === bandId) {
      setDragOverBand(null);
    }
  };

  const handleDrop = (e, bandId) => {
    e.preventDefault();
    setDragOverBand(null);
    const droppedFile = e.dataTransfer.files?.[0];
    if (droppedFile) {
      onSelectFile(bandId, droppedFile);
    }
  };

  return (
    <>
      <div className="page-eyebrow">DATA INPUT / MULTISPECTRAL SATELLITE</div>

      <h1>
        Multispectral <span className="gradient-text">processing.</span>
      </h1>

      <p className="page-description">
        Upload 4 satellite spectral bands (Red, Green, Blue, and Near-Infrared) to run deep learning cloud segmentation.
      </p>

      <form className="upload-panel" onSubmit={onSubmit}>
        <div className="panel-heading">
          <div>
            <div className="page-eyebrow">INPUT CONFIGURATION</div>
            <h2>Satellite Band Uploads</h2>
          </div>

          <div className="status-badge-group">
            <span className={`required-tag ${allFilesUploaded ? "tag-complete" : ""}`}>
              {uploadedCount} / 4 BANDS SELECTED
            </span>
          </div>
        </div>

        <div className="band-grid">
          {BANDS.map((band, index) => {
            const file = files[band.id];
            const isDragged = dragOverBand === band.id;

            return (
              <div
                key={band.id}
                className={`band-card ${file ? "has-file" : ""} ${isDragged ? "drag-active" : ""}`}
                style={{ "--band-color": band.color }}
                onDragOver={(e) => handleDragOver(e, band.id)}
                onDragLeave={(e) => handleDragLeave(e, band.id)}
                onDrop={(e) => handleDrop(e, band.id)}
                onClick={() => fileInputRefs.current[band.id]?.click()}
              >
                <div className="band-card-top">
                  <span className="band-number">0{index + 1}</span>
                  <span className="band-badge">{band.badge}</span>
                  <span className={`file-status-pill ${file ? "ready" : "empty"}`}>
                    {file ? "READY" : "WAITING"}
                  </span>
                </div>

                <div className="band-letter">{band.letter}</div>

                <h3>{band.name}</h3>
                <p>{band.description}</p>

                <div className="choose-file">
                  {file ? (
                    <>
                      <CheckCircle2 size={16} /> File Selected
                    </>
                  ) : (
                    <>
                      <UploadCloud size={16} /> Drop or Browse
                    </>
                  )}
                </div>

                <input
                  ref={(el) => (fileInputRefs.current[band.id] = el)}
                  type="file"
                  style={{ display: "none" }}
                  accept=".tif,.tiff,.png,.jpg,.jpeg"
                  onChange={(event) => {
                    const selected = event.target.files?.[0];
                    if (selected) onSelectFile(band.id, selected);
                  }}
                />

                {file && (
                  <div className="file-info-bar">
                    <span className="file-name" title={file.name}>{file.name}</span>
                    <span className="file-size">{(file.size / 1024).toFixed(0)} KB</span>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Hyperparameters Controls */}
        <div className="hyperparameters-bar">
          <div className="control-group">
            <div className="control-header">
              <Sliders size={16} />
              <span>Confidence Threshold: <b>{Math.round(threshold * 100)}%</b></span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="range-slider"
            />
          </div>

          <div className="control-group">
            <div className="control-header">
              <Sliders size={16} />
              <span>Overlay Opacity: <b>{Math.round(opacity * 100)}%</b></span>
            </div>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={opacity}
              onChange={(e) => setOpacity(parseFloat(e.target.value))}
              className="range-slider"
            />
          </div>
        </div>

        <p className="upload-note">
          Supports Landsat 8 (38-Cloud dataset) & Sentinel-2 formats: TIFF, GeoTIFF, PNG, and JPEG. Files are validated and processed in-memory for zero disk leakage.
        </p>

        {error && (
          <div className="error-message">
            <AlertTriangle size={18} />
            <span>{error}</span>
          </div>
        )}

        <div className="form-actions">
          <button
            type="button"
            className="secondary-button"
            onClick={onResetFiles}
            disabled={loading || uploadedCount === 0}
          >
            <RotateCcw size={16} />
            Reset Bands
          </button>

          <button
            type="submit"
            className="launch-button"
            disabled={loading || !allFilesUploaded}
          >
            {loading ? (
              <>
                <span className="button-spinner" />
                Analyzing Satellite Bands...
              </>
            ) : (
              <>
                Run Cloud Segmentation
                <ArrowRight size={18} />
              </>
            )}
          </button>
        </div>
      </form>
    </>
  );
}
