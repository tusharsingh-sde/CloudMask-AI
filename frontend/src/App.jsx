import { useState, useEffect, useRef } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import Navbar from "./components/Navbar";
import HeroSection from "./components/HeroSection";
import ProcessingDashboard from "./components/ProcessingDashboard";
import ResultViewer from "./components/ResultViewer";
import { checkBackendHealth, predictCloudMask } from "./services/api";
import "./App.css";

gsap.registerPlugin(ScrollTrigger);

export default function App() {
  const [activeSection, setActiveSection] = useState("overview");
  const [files, setFiles] = useState({});
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [threshold, setThreshold] = useState(0.5);
  const [opacity, setOpacity] = useState(0.45);
  const [backendHealth, setBackendHealth] = useState(null);

  const mainContainerRef = useRef(null);

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

  // GSAP Animations and ScrollTriggers (React 19 StrictMode safe)
  useEffect(() => {
    const ctx = gsap.context(() => {
      // 1. Hero Entrance Animation: animate subtle y offset ONLY so opacity is NEVER locked to 0
      const heroTl = gsap.timeline({ defaults: { ease: "power2.out" } });
      heroTl
        .fromTo(
          ".hero-eyebrow-line",
          { y: -12 },
          { y: 0, duration: 0.6, delay: 0.1, clearProps: "all" }
        )
        .fromTo(
          ".hero-title",
          { y: 18 },
          { y: 0, duration: 0.7, clearProps: "all" },
          "-=0.4"
        )
        .fromTo(
          ".hero-description",
          { y: 14 },
          { y: 0, duration: 0.55, clearProps: "all" },
          "-=0.4"
        )
        .fromTo(
          "#hero-launch-btn",
          { y: 14 },
          { y: 0, duration: 0.5, clearProps: "all" },
          "-=0.3"
        )
        .fromTo(
          ".telemetry-chip",
          { y: 10 },
          { y: 0, stagger: 0.08, duration: 0.4, clearProps: "all" },
          "-=0.3"
        );

      // 2. Parallax and soft fade on hero 3D canvas as user scrolls into workspace
      gsap.to(".hero-webgl-canvas-container", {
        scrollTrigger: {
          trigger: "#overview",
          start: "top top",
          end: "bottom top",
          scrub: true,
        },
        y: 120,
        opacity: 0.35,
        ease: "none",
      });

      // 3. Stagger reveal workspace cards on scroll (guaranteed completion, no opacity zeroing)
      gsap.fromTo(
        ".band-card",
        { y: 25 },
        {
          scrollTrigger: {
            trigger: "#workspace",
            start: "top 92%",
            once: true,
          },
          y: 0,
          stagger: 0.08,
          duration: 0.55,
          ease: "power2.out",
          clearProps: "all",
        }
      );

      // 4. Track Active Section on Scroll
      const sections = ["overview", "workspace", "results"];
      sections.forEach((secId) => {
        ScrollTrigger.create({
          trigger: `#${secId}`,
          start: "top center",
          end: "bottom center",
          onEnter: () => setActiveSection(secId),
          onEnterBack: () => setActiveSection(secId),
        });
      });

      // 5. Ensure coordinates are up-to-date
      ScrollTrigger.refresh();
    }, mainContainerRef);

    return () => ctx.revert();
  }, []);

  const scrollToSection = (sectionId) => {
    setActiveSection(sectionId);
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
    }
  };

  const handleSelectFile = (id, file) => {
    setFiles((prev) => ({
      ...prev,
      [id]: file,
    }));
    setError("");
  };

  const handleResetFiles = () => {
    setFiles({});
    setResult(null);
    setError("");
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
      setResult(output);

      // Smoothly scroll to results section
      setTimeout(() => {
        const resultsEl = document.getElementById("results");
        if (resultsEl) {
          resultsEl.scrollIntoView({ behavior: "smooth" });
          setActiveSection("results");
        }
      }, 120);
    } catch (err) {
      setError(err.message || "Failed to process the satellite bands.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="aerospace-app" ref={mainContainerRef}>
      {/* Sticky Minimal Aerospace Top Navbar */}
      <Navbar
        activeSection={activeSection}
        onNavigate={scrollToSection}
        backendHealth={backendHealth}
      />

      <main className="aerospace-main-content">
        {/* Section 1: Hero Overview with 3D Earth */}
        <HeroSection onLaunchWorkspace={() => scrollToSection("workspace")} />

        {/* Section 2: Multispectral Processing Workspace */}
        <section id="workspace" className="aerospace-section workspace-section">
          <div className="section-container">
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
          </div>
        </section>

        {/* Section 3: Deep Learning Analysis Results */}
        <section id="results" className="aerospace-section results-section">
          <div className="section-container">
            <ResultViewer
              result={result}
              onNewAnalysis={() => scrollToSection("workspace")}
            />
          </div>
        </section>
      </main>

      {/* Sleek Aerospace Mission-Control Footer */}
      <footer className="aerospace-footer">
        <div className="footer-container">
          <div className="footer-left">
            <span className="footer-brand">CLOUDMASK AI</span>
            <span className="footer-sep">·</span>
            <span className="footer-sub">LANDSAT 8 OLI / TIRS MULTISPECTRAL ENGINE</span>
          </div>
          <div className="footer-right">
            <span className="footer-telemetry">EPSG:4326 · 38-CLOUD U-NET ARCHITECTURE</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
