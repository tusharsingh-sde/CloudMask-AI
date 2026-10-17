import EarthScene from "./EarthScene";
import { ArrowRight, Play, Satellite, Activity } from "lucide-react";

export default function LandingPage({ onLaunch, backendHealth }) {
  const isOnline = backendHealth?.status === "online";

  return (
    <main className="landing">
      <div className="space-background">
        <div className="star-field star-field-one" />
        <div className="star-field star-field-two" />
        <div className="milky-way" />
        <div className="nebula-cloud nebula-cloud-one" />
        <div className="nebula-cloud nebula-cloud-two" />
        <div className="galaxy galaxy-one" />
        <div className="galaxy galaxy-two" />
      </div>

      <header className="landing-header">
        <div className="wordmark">
          <span className="cloud-logo">
            <span />
          </span>
          <span>
            CloudMask <b>AI</b>
          </span>
        </div>

        <div className="landing-status-pill">
          <span className={`status-dot ${isOnline ? "dot-online" : "dot-checking"}`} />
          <span>{isOnline ? "NEURAL CORE ACTIVE" : "LOCAL BACKEND READY"}</span>
        </div>
      </header>

      <section className="landing-content">
        <div className="landing-copy">
          <div className="eyebrow">
            <Activity size={14} className="pulse-icon" />
            REAL-TIME EARTH MONITORING
          </div>

          <h1>
            See Earth in a
            <span className="gradient-text"> Different Spectrum</span>
          </h1>

          <p>
            From visible light to thermal, infrared and beyond — explore our planet
            with multispectral satellite band segmentation powered by deep neural networks.
          </p>

          <div className="landing-action-group">
            <button className="launch-button" onClick={onLaunch}>
              <Play size={16} fill="currentColor" className="play-icon" />
              Launch Dashboard
              <ArrowRight size={18} className="launch-arrow" />
            </button>

            <div className="landing-metric">
              <Satellite size={16} />
              <span>4 Multispectral Bands (R, G, B, NIR)</span>
            </div>
          </div>
        </div>

        <div className="space-scene">
          <EarthScene />

          <div className="scene-label scene-label-top">
            <i />
            LIVE ORBITAL NETWORK
          </div>

          <div className="scene-label scene-label-bottom">
            MULTISPECTRAL IMAGING
            <span />
          </div>
        </div>
      </section>
    </main>
  );
}
