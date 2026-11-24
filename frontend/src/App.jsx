import { useState, useEffect } from "react";
import Navbar from "./components/Navbar";
import HeroSection from "./components/HeroSection";
import StudioWorkbench from "./components/StudioWorkbench";
import { checkBackendHealth, predictCloudMask } from "./services/api";
import "./App.css";

export default function App() {
  const [activeSection, setActiveSection] = useState("overview");
  const [files, setFiles] = useState({});
  const [result, setResult] = useState(null);
  const [runHistory, setRunHistory] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [threshold, setThreshold] = useState(0.5);
  const [opacity, setOpacity] = useState(0.45);
  const [backendHealth, setBackendHealth] = useState(null);

  // Poll backend health on initial load and every 30s
  useEffect(() => {
    let isMounted = true;
    async function verifyBackend() {
      const health = await checkBackendHealth();
      if (isMounted) {
        setBackendHealth(health);
      }
    }
    verifyBackend();

    const interval = setInterval(verifyBackend, 30000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  // Track active section using IntersectionObserver (native, lightweight, zero opacity tampering)
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveSection(entry.target.id);
          }
        });
      },
      { threshold: 0.35 }
    );

    const sections = ["overview", "studio"];
    sections.forEach((id) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });

    return () => observer.disconnect();
  }, []);

  const scrollToSection = (sectionId) => {
    setActiveSection(sectionId);
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
    }
  };

  const handleFilesChange = (updatedFiles) => {
    setFiles(updatedFiles);
    setError("");
  };

  const handleResetFiles = () => {
    setFiles({});
    setResult(null);
    setError("");
  };

  const handleSelectRun = (historyItem) => {
    setResult(historyItem);
  };

  const handleProcessBands = async (event) => {
    if (event) event.preventDefault();
    setError("");

    try {
      setLoading(true);
      const output = await predictCloudMask(files, {
        threshold,
        overlayOpacity: opacity,
      });

      const enrichedOutput = {
        ...output,
        id: `run-${Date.now()}`,
        timestamp: new Date().toLocaleTimeString(),
      };

      setResult(enrichedOutput);
      setRunHistory((prev) => [enrichedOutput, ...prev.slice(0, 5)]);

      // Ensure view is on studio workbench
      const studioEl = document.getElementById("studio");
      if (studioEl) {
        studioEl.scrollIntoView({ behavior: "smooth" });
        setActiveSection("studio");
      }
    } catch (err) {
      setError(err.message || "Failed to process the satellite bands.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="ground-station-app">
      {/* Sticky Glass Topbar (60px) */}
      <Navbar
        activeSection={activeSection}
        onNavigate={scrollToSection}
        backendHealth={backendHealth}
      />

      <main className="app-main-content">
        {/* Section 1: Hero Overview with 3D Earth */}
        <HeroSection onOpenStudio={() => scrollToSection("studio")} />

        {/* Section 2: Unified Ground-Station Studio Workbench */}
        <StudioWorkbench
          files={files}
          onFilesChange={handleFilesChange}
          onResetFiles={handleResetFiles}
          onSubmit={handleProcessBands}
          loading={loading}
          error={error}
          threshold={threshold}
          setThreshold={setThreshold}
          opacity={opacity}
          setOpacity={setOpacity}
          result={result}
          runHistory={runHistory}
          onSelectRun={handleSelectRun}
          backendHealth={backendHealth}
        />
      </main>

      {/* Minimal Mission Control Footer */}
      <footer className="ground-station-footer">
        <div className="footer-inner">
          <div className="footer-left">
            <span className="footer-title">CLOUDMASK AI</span>
            <span className="footer-sep">/</span>
            <span className="footer-desc">
              SATELLITE GROUND STATION & EARTH OBSERVATION WORKBENCH
            </span>
          </div>
          <div className="footer-right">
            <span>LANDSAT 8 OLI · EPSG:4326 · 38-CLOUD DATASET</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
