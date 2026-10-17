import { Home, Layers, PieChart, ShieldCheck, AlertCircle } from "lucide-react";

export default function Sidebar({ currentPage, onNavigate, backendHealth }) {
  const isOnline = backendHealth?.status === "online";
  const device = backendHealth?.device || "cpu";

  return (
    <aside className="sidebar">
      <button className="wordmark dashboard-brand" onClick={() => onNavigate("landing")}>
        <span className="cloud-logo">
          <span />
        </span>
        <span>
          CloudMask <b>AI</b>
        </span>
      </button>

      <p className="side-label">WORKSPACE</p>

      <nav className="side-nav">
        <button
          className={currentPage === "landing" ? "selected" : ""}
          onClick={() => onNavigate("landing")}
        >
          <Home size={18} />
          Home
        </button>

        <button
          className={currentPage === "processing" ? "selected" : ""}
          onClick={() => onNavigate("processing")}
        >
          <Layers size={18} />
          Processing
        </button>

        <button
          className={currentPage === "result" ? "selected" : ""}
          onClick={() => onNavigate("result")}
        >
          <PieChart size={18} />
          Result
        </button>
      </nav>

      <div className="sidebar-bottom">
        <div className="backend-status-card">
          <div className="status-row">
            {isOnline ? (
              <>
                <ShieldCheck size={16} className="text-emerald" />
                <span className="status-text text-emerald">BACKEND ONLINE</span>
              </>
            ) : (
              <>
                <AlertCircle size={16} className="text-amber" />
                <span className="status-text text-amber">BACKEND OFFLINE</span>
              </>
            )}
          </div>
          <small className="status-sub">
            {isOnline ? `PyTorch ${device.toUpperCase()} · U-Net Ready` : "Waiting for localhost:8000"}
          </small>
        </div>
      </div>
    </aside>
  );
}
