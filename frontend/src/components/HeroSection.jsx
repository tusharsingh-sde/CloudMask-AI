import { ArrowRight, ChevronDown, Cpu, Layers } from "lucide-react";
import EarthScene from "./EarthScene";

export default function HeroSection({ onOpenStudio }) {
  return (
    <section id="overview" className="hero-section">
      <div className="hero-webgl-canvas-container">
        <EarthScene />
      </div>

      <div className="hero-inner">
        <div className="hero-copy-column">
          <div className="hero-eyebrow">
            <span className="eyebrow-line" />
            <span className="eyebrow-text">LANDSAT 8 · 38-CLOUD U-NET</span>
          </div>

          <h1 className="hero-headline">
            <span className="headline-line-white">See through the clouds.</span>
            <span className="headline-line-gradient">Resolve every pixel.</span>
          </h1>

          <p className="hero-pitch">
            Radiometric deep learning pipeline isolating sub-kilometer cloud formations and shadows across 4 spectral bands in real time with zero disk latency.
          </p>

          <div className="hero-cta-row">
            <button
              className="hero-primary-cta"
              onClick={onOpenStudio}
              id="hero-launch-btn"
            >
              <span>Open Segmentation Studio</span>
              <ArrowRight size={16} className="cta-arrow-icon" />
            </button>

            <div className="hero-specs-badge">
              <span className="specs-dot" />
              <span>4 Bands · 384×384 · PyTorch</span>
            </div>
          </div>

          <div className="hero-telemetry-row">
            <div className="telemetry-pill">
              <Layers size={13} className="pill-icon" />
              <span>OLI B02–B05 SPECTRAL ARRAYS</span>
            </div>
            <div className="telemetry-pill">
              <Cpu size={13} className="pill-icon" />
              <span>38-CLOUD WEIGHTS</span>
            </div>
          </div>
        </div>
      </div>

      <div className="hero-bottom-bar">
        <div className="hero-bottom-left">
          <span className="meta-sync-dot" />
          <span className="meta-text">OLI BANDS B02–B05 · IN-MEMORY PIPELINE</span>
        </div>

        <button
          className="hero-scroll-cue"
          onClick={onOpenStudio}
          aria-label="Scroll to Segmentation Studio"
        >
          <span>SCROLL TO STUDIO</span>
          <ChevronDown size={14} className="cue-chevron" />
        </button>
      </div>
    </section>
  );
}
