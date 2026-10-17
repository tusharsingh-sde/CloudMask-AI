import { useState, useEffect } from "react";
import LandingPage from "./components/LandingPage";
import Sidebar from "./components/Sidebar";
import ProcessingDashboard from "./components/ProcessingDashboard";
import ResultViewer from "./components/ResultViewer";
import { checkBackendHealth, predictCloudMask } from "./services/api";
import "./App.css";

export default function App() {
  const [page, setPage] = useState("landing");
  const [files, setFiles] = useState({});
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [threshold, setThreshold] = useState(0.5);
  const [opacity, setOpacity] = useState(0.45);
  const [backendHealth, setBackendHealth] = useState(null);

  // Poll backend health on initial load
  useEffect(() => {
    let isMounted = true;
    async function verifyBackend() {
      const health = await checkBackendHealth();
      if (isMounted) {
        setBackendHealth(health);
      }
    }
    verifyBackend();

    // Recheck health every 30 seconds
    const interval = setInterval(verifyBackend, 30000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  function handleSelectFile(id, file) {
    setFiles((prev) => ({
      ...prev,
      [id]: file,
    }));
    setError("");
  }

  function handleResetFiles() {
    setFiles({});
    setResult(null);
    setError("");
  }

  async function handleProcessBands(event) {
    if (event) event.preventDefault();
    setError("");
    setResult(null);

    try {
      setLoading(true);
      const output = await predictCloudMask(files, {
        threshold,
        overlayOpacity: opacity,
      });
      setResult(output);
      setPage("result");
    } catch (err) {
      setError(err.message || "Failed to process the satellite bands.");
    } finally {
      setLoading(false);
    }
  }

  if (page === "landing") {
    return (
      <LandingPage
        onLaunch={() => {
          setPage("processing");
          setError("");
        }}
        backendHealth={backendHealth}
      />
    );
  }

  return (
    <div className="dashboard">
      <Sidebar
        currentPage={page}
        onNavigate={(target) => {
          setPage(target);
          setError("");
        }}
        backendHealth={backendHealth}
      />

      <section className="dashboard-main">
        <header className="dashboard-topbar">
          <div className="topbar-breadcrumbs">
            <span>Workspace</span>
            <span className="sep">/</span>
            <span className="current-crumb">
              {page === "processing" ? "Processing" : "Analysis Result"}
            </span>
          </div>

          <div className="topbar-actions">
            <span className="top-status">
              <span className={`status-dot ${backendHealth?.status === "online" ? "dot-online" : "dot-offline"}`} />
              {backendHealth?.status === "online" ? "NEURAL CORE CONNECTED" : "OFFLINE / CONNECTING"}
            </span>
          </div>
        </header>

        <div className="dashboard-content">
          {page === "processing" && (
            <ProcessingDashboard
              files={files}
              onSelectFile={handleSelectFile}
              onResetFiles={handleResetFiles}
              onSubmit={handleProcessBands}
              loading={loading}
              error={error}
              threshold={threshold}
              setThreshold={setThreshold}
              opacity={opacity}
              setOpacity={setOpacity}
            />
          )}

          {page === "result" && (
            <ResultViewer
              result={result}
              onNewAnalysis={() => {
                setPage("processing");
                setError("");
              }}
            />
          )}
        </div>
      </section>
    </div>
  );
}