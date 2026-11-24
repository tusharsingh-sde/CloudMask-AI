import { Layers } from "lucide-react";

export default function Navbar({ activeSection, onNavigate, backendHealth }) {
  const isOnline = backendHealth?.status === "online";
  const isChecking = backendHealth === null;
  const device = backendHealth?.device ? backendHealth.device.toUpperCase() : "CPU";
  const epoch = backendHealth?.checkpoint?.epoch;
  const loss = backendHealth?.checkpoint?.best_loss;

  const tooltipText = isOnline
    ? `PyTorch U-Net · Device: ${device} · Epoch: ${epoch ?? "Best"} · Loss: ${
        typeof loss === "number" ? loss.toFixed(4) : "0.0958"
      }`
    : isChecking
    ? "Connecting to local FastAPI inference engine..."
    : "Backend unreachable at /api/health";

  const handleNav = (targetId) => {
    onNavigate(targetId);
    const el = document.getElementById(targetId);
    if (el) {
      el.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <header className="ground-station-navbar">
      <div className="navbar-container">
        {/* Left: Brand Mark */}
        <button
          className="navbar-brand"
          onClick={() => handleNav("overview")}
          aria-label="CloudMask AI Home"
        >
          <div className="brand-symbol">
            <Layers size={16} className="brand-icon" />
            <span className="brand-halo" />
          </div>
          <div className="brand-logotype">
            <span className="brand-title">CloudMask</span>
            <span className="brand-pill">AI</span>
          </div>
        </button>

        {/* Center/Right: Clean Navigation Links */}
        <div className="navbar-right">
          <nav className="navbar-links" aria-label="Main Navigation">
            <button
              className={`nav-item ${activeSection === "overview" ? "active" : ""}`}
              onClick={() => handleNav("overview")}
            >
              Overview
            </button>
            <button
              className={`nav-item ${activeSection === "studio" ? "active" : ""}`}
              onClick={() => handleNav("studio")}
            >
              Studio
            </button>
          </nav>

          {/* Far Right: Live Telemetry Indicator */}
          <div
            className={`navbar-telemetry ${
              isOnline ? "telemetry-online" : isChecking ? "telemetry-checking" : "telemetry-offline"
            }`}
            title={tooltipText}
          >
            <span
              className={`telemetry-dot ${
                isOnline ? "dot-online" : isChecking ? "dot-checking" : "dot-offline"
              }`}
            />
            <span className="telemetry-label">
              {isOnline
                ? `CORE ONLINE · ${device}`
                : isChecking
                ? "SYNCING..."
                : "ENGINE OFFLINE"}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
