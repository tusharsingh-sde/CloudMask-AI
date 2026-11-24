import { ArrowRight, ChevronDown, Satellite, Cpu, Layers } from "lucide-react";
import EarthScene from "./EarthScene";

export default function HeroSection({ onLaunchWorkspace }) {
  return (
    <section id="overview" className="hero-section">
      {/* Full-bleed 3D WebGL Background Canvas */}
      <div className="hero-webgl-canvas-container">
        <EarthScene />
      </div>

      {/* Hero Foreground Content */}
      <div className="hero-inner">
        <div className="hero-copy-column">
          {/* Minimal Eyebrow: 28px x 1px cyan line + tracked mono label */}
          <div className="hero-eyebrow-line">
            <span className="eyebrow-accent-bar" />
            <span className="eyebrow-mono-text">LANDSAT 8 · MULTISPECTRAL U-NET ENGINE</span>
          </div>

          {/* Crisp Light-Weight 2-Line Editorial Headline */}
          <h1 className="hero-title">
            <span className="title-line-white">Clouds obscure the planet.</span>
            <span className="title-line-cyan">We isolate every pixel.</span>
          </h1>

          {/* Concise Technical Description */}
          <p className="hero-description">
            Sub-pixel multispectral segmentation using synchronized 4-band spectral radiance (Red, Green, Blue, and Near-Infrared) to eliminate atmospheric obscuration and resolve ground radiance.
          </p>

          {/* Clean, Always-Visible Aerospace CTA Button */}
          <div className="hero-cta-group">
            <button
              className="aerospace-cta-btn"
              onClick={onLaunchWorkspace}
              id="hero-launch-btn"
            >
              <span className="cta-dot" />
              <span className="cta-label">BEGIN SEGMENTATION</span>
              <ArrowRight size={15} className="cta-arrow" />
            </button>
          </div>

          {/* 3 Compact Monospace Telemetry Chips */}
          <div className="hero-telemetry-chips">
            <div className="telemetry-chip">
              <Satellite size={13} className="chip-icon" />
              <span>4 SPECTRAL BANDS</span>
            </div>
            <div className="telemetry-chip">
              <Cpu size={13} className="chip-icon" />
              <span>U-NET 38-CLOUD</span>
            </div>
            <div className="telemetry-chip">
              <Layers size={13} className="chip-icon" />
              <span>IN-MEMORY INFERENCE</span>
            </div>
          </div>
        </div>
      </div>

      {/* Hero Bottom Telemetry & Scroll Indicator */}
      <div className="hero-bottom-bar">
        <div className="hero-orbital-telemetry">
          <span className="live-coord-pulse" />
          <span className="mono-readout">ORBITAL SYNC · 384×384 PATCHES</span>
        </div>

        <button
          className="hero-scroll-btn"
          onClick={onLaunchWorkspace}
          aria-label="Scroll to Workspace"
        >
          <span className="scroll-text">SCROLL</span>
          <ChevronDown size={14} className="scroll-chevron" />
        </button>
      </div>
    </section>
  );
}
