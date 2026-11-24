import { Satellite } from "lucide-react";

export default function Navbar({ activeSection, onNavigate, backendHealth }) {
  const isOnline = backendHealth?.status === "online";
  const device = backendHealth?.device ? backendHealth.device.toUpperCase() : "CPU";

  const handleNavClick = (sectionId) => {
    onNavigate(sectionId);
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <header className="aerospace-navbar">
      <div className="nav-container">
        {/* Brand Mark */}
        <button
          className="nav-brand"
          onClick={() => handleNavClick("overview")}
          aria-label="CloudMask AI Overview"
        >
          <div className="brand-icon-wrapper">
            <Satellite size={16} className="brand-satellite-icon" />
            <span className="brand-orbit-ring" />
          </div>
          <div className="brand-text">
            <span className="brand-name">CLOUDMASK</span>
            <span className="brand-tag">AI</span>
          </div>
        </button>

        {/* Minimal Nav Links + Minimal Telemetry Indicator */}
        <div className="nav-actions">
          <nav className="nav-links" aria-label="Main Navigation">
            <button
              className={`nav-link-btn ${activeSection === "overview" ? "active" : ""}`}
              onClick={() => handleNavClick("overview")}
            >
              OVERVIEW
            </button>
            <button
              className={`nav-link-btn ${activeSection === "workspace" ? "active" : ""}`}
              onClick={() => handleNavClick("workspace")}
            >
              WORKSPACE
            </button>
            <button
              className={`nav-link-btn ${activeSection === "results" ? "active" : ""}`}
              onClick={() => handleNavClick("results")}
            >
              RESULTS
            </button>
          </nav>

          {/* Minimal Sleek Status Readout (No Bulky Box) */}
          <div className="nav-minimal-status">
            <span className={`status-indicator-dot ${isOnline ? "dot-emerald" : "dot-rose"}`} />
            <span className="minimal-status-text">
              {isOnline ? `NEURAL CORE ONLINE · ${device}` : "BACKEND OFFLINE"}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
