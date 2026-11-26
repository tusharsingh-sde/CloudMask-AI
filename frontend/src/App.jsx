import { useState, useEffect } from "react";
import { Sparkles, ChevronRight, ChevronLeft, X, Snowflake, CheckCircle2 } from "lucide-react";
import Navbar from "./components/Navbar";
import HeroSection from "./components/HeroSection";
import StudioWorkbench from "./components/StudioWorkbench";
import { SAMPLE_SCENES } from "./constants/samples";
import { checkBackendHealth, predictCloudMask } from "./services/api";
import "./App.css";

const DEMO_STEPS = [
  {
    step: 1,
    badge: "STEP 1 OF 3 · INGESTION",
    title: "4 Synchronized 16-bit Landsat-8 Bands Loaded",
    body: "Red (B04), Green (B03), Blue (B02), and Near-Infrared (B05) GeoTIFF bands were auto-loaded and passed through the U-Net segmentation pipeline.",
  },
  {
    step: 2,
    badge: "STEP 2 OF 3 · INSPECTION",
    title: "Drag the Cyan Curtain to Compare Mask vs. RGB",
    body: "Slide the vertical divider left/right (or use ← / → keys). You can also switch to Binary Mask, Probability Heatmap, or Quad Grid above the viewport.",
  },
  {
    step: 3,
    badge: "STEP 3 OF 3 · NIR STRESS TEST",
    title: "Test Snow vs. Cloud Rejection (Band 5 NIR)",
    body: "Normal RGB models mistake bright snow for clouds. Click below to run a snow-covered alpine scene and watch the 4-band U-Net keep false positives under 1%.",
  },
];

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

  // Shared sample & guided demo state
  const [activeSampleId, setActiveSampleId] = useState(null);
  const [loadingSampleId, setLoadingSampleId] = useState(null);
  const [demoStep, setDemoStep] = useState(0); // 0 = closed, 1..3 = active step

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

  // Track active section using IntersectionObserver
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
    setActiveSampleId(null);
  };

  const handleSelectRun = (historyItem) => {
    setResult(historyItem);
  };

  const handleProcessBands = async (eventOrFiles) => {
    let targetFiles = files;
    if (eventOrFiles && typeof eventOrFiles.preventDefault === "function") {
      eventOrFiles.preventDefault();
    } else if (
      eventOrFiles &&
      typeof eventOrFiles === "object" &&
      !eventOrFiles.nativeEvent &&
      !eventOrFiles.target
    ) {
      targetFiles = eventOrFiles;
    }
    setError("");

    try {
      setLoading(true);
      const output = await predictCloudMask(targetFiles, {
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

      const studioEl = document.getElementById("studio");
      if (studioEl) {
        studioEl.scrollIntoView({ behavior: "smooth" });
        setActiveSection("studio");
      }
      return true;
    } catch (err) {
      setError(err.message || "Failed to process the satellite bands.");
      return false;
    } finally {
      setLoading(false);
    }
  };

  // Unified 1-click sample loader (used by Sidebar cards + Navbar Live Demo + Coachbar)
  const handleLoadSample = async (sceneOrId) => {
    if (loading || loadingSampleId !== null) return false;
    const scene =
      typeof sceneOrId === "string"
        ? SAMPLE_SCENES.find((s) => s.id === sceneOrId)
        : sceneOrId;
    if (!scene) return false;

    setLoadingSampleId(scene.id);
    setError("");

    try {
      const baseUrl = (import.meta.env.BASE_URL || "/").replace(/\/$/, "");
      const bandNames = ["red", "green", "blue", "nir"];

      const results = await Promise.all(
        bandNames.map(async (band) => {
          const url = `${baseUrl}/samples/${scene.folder}/${band}.tif`;
          const res = await fetch(url);
          if (!res.ok) {
            throw new Error(`Failed to load ${band} band for ${scene.title} (HTTP ${res.status})`);
          }
          const blob = await res.blob();
          return {
            band,
            file: new File([blob], `${band}_${scene.patchSuffix}.tif`, {
              type: "image/tiff",
              lastModified: Date.now(),
            }),
          };
        })
      );

      const loadedFiles = {};
      results.forEach(({ band, file }) => {
        loadedFiles[band] = file;
      });

      setActiveSampleId(scene.id);
      setFiles(loadedFiles);
      return await handleProcessBands(loadedFiles);
    } catch (err) {
      setActiveSampleId(null);
      setError(err.message || "Failed to load sample scene.");
      return false;
    } finally {
      setLoadingSampleId(null);
    }
  };

  // Topbar "Live Demo" CTA handler
  const handleStartDemo = async () => {
    scrollToSection("studio");
    setDemoStep(1);
    await handleLoadSample("partial-cloud");
  };

  const currentDemo = demoStep > 0 ? DEMO_STEPS[demoStep - 1] : null;

  return (
    <div className="ground-station-app">
      <Navbar
        activeSection={activeSection}
        onNavigate={scrollToSection}
        backendHealth={backendHealth}
        onStartDemo={handleStartDemo}
        demoLoading={loading || loadingSampleId !== null}
      />

      <main className="app-main-content">
        <HeroSection onOpenStudio={() => scrollToSection("studio")} />

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
          activeSampleId={activeSampleId}
          loadingSampleId={loadingSampleId}
          onLoadSample={handleLoadSample}
          onClearActiveSample={() => setActiveSampleId(null)}
        />
      </main>

      {/* Guided Demo Coachbar */}
      {currentDemo && (
        <aside className="demo-coachbar" role="dialog" aria-label="Interactive Demo Guide">
          <div className="coachbar-top">
            <span className="coachbar-badge">
              <Sparkles size={12} />
              <span>{currentDemo.badge}</span>
            </span>
            <button
              type="button"
              className="coachbar-close"
              onClick={() => setDemoStep(0)}
              aria-label="Close demo guide"
            >
              <X size={14} />
            </button>
          </div>

          <h4 className="coachbar-title">{currentDemo.title}</h4>
          <p className="coachbar-body">
            {loading && demoStep === 1
              ? "Fetching 4 GeoTIFF bands and running live PyTorch U-Net inference..."
              : currentDemo.body}
          </p>

          <div className="coachbar-actions">
            <div className="coachbar-dots">
              {DEMO_STEPS.map((s) => (
                <span
                  key={s.step}
                  className={`coachbar-dot ${demoStep === s.step ? "active" : ""}`}
                />
              ))}
            </div>

            <div className="coachbar-btns">
              {demoStep > 1 && (
                <button
                  type="button"
                  className="coachbar-btn-ghost"
                  onClick={() => setDemoStep((s) => s - 1)}
                >
                  <ChevronLeft size={14} />
                  <span>Back</span>
                </button>
              )}

              {demoStep === 3 && (
                <button
                  type="button"
                  className="coachbar-btn-accent"
                  disabled={loading || loadingSampleId !== null}
                  onClick={() => handleLoadSample("snow-vs-cloud")}
                >
                  <Snowflake size={13} />
                  <span>
                    {activeSampleId === "snow-vs-cloud" && !loading
                      ? "Snow Scene Loaded"
                      : "Run Snow Test"}
                  </span>
                </button>
              )}

              {demoStep < 3 ? (
                <button
                  type="button"
                  className="coachbar-btn-primary"
                  disabled={loading}
                  onClick={() => setDemoStep((s) => s + 1)}
                >
                  <span>Next</span>
                  <ChevronRight size={14} />
                </button>
              ) : (
                <button
                  type="button"
                  className="coachbar-btn-primary"
                  onClick={() => setDemoStep(0)}
                >
                  <CheckCircle2 size={14} />
                  <span>Done</span>
                </button>
              )}
            </div>
          </div>
        </aside>
      )}

      {/* Mission Control Footer */}
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
