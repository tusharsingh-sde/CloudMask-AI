import { useEffect, useRef } from "react";
import * as THREE from "three";

const EARTH_TEXTURE =
  "https://threejs.org/examples/textures/planets/earth_atmos_2048.jpg";
const EARTH_NORMAL =
  "https://threejs.org/examples/textures/planets/earth_normal_2048.jpg";
const EARTH_SPECULAR =
  "https://threejs.org/examples/textures/planets/earth_specular_2048.jpg";
const EARTH_CLOUDS =
  "https://threejs.org/examples/textures/planets/earth_clouds_1024.png";

// Programmatic Atmospheric Halo CanvasTexture (Radial Gradient: 0.42 -> 0.24 -> 0.08 -> 0)
function createAtmosphereHaloTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext("2d");
  if (!ctx) return new THREE.Texture();

  const grad = ctx.createRadialGradient(256, 256, 0, 256, 256, 256);
  grad.addColorStop(0.0, "rgba(56, 189, 248, 0.42)");
  grad.addColorStop(0.38, "rgba(14, 165, 233, 0.24)");
  grad.addColorStop(0.62, "rgba(29, 78, 216, 0.08)");
  grad.addColorStop(1.0, "rgba(0, 0, 0, 0.0)");

  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 512, 512);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

// Particle Dot Texture for 3D Wave
function createParticleDotTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext("2d");
  if (!ctx) return new THREE.Texture();

  const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0.0, "rgba(255, 255, 255, 1.0)");
  grad.addColorStop(0.25, "rgba(56, 189, 248, 0.9)");
  grad.addColorStop(0.65, "rgba(37, 99, 235, 0.35)");
  grad.addColorStop(1.0, "rgba(0, 0, 0, 0.0)");

  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 64, 64);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

// Fallback texture if remote planet images are blocked
function createProceduralEarthFallback() {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 256;
  const ctx = canvas.getContext("2d");
  if (!ctx) return new THREE.Texture();

  const grad = ctx.createLinearGradient(0, 0, 512, 256);
  grad.addColorStop(0, "#081d3d");
  grad.addColorStop(0.4, "#0e3a6c");
  grad.addColorStop(0.8, "#144a82");
  grad.addColorStop(1, "#0a264a");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 512, 256);

  ctx.fillStyle = "rgba(56, 189, 248, 0.22)";
  ctx.beginPath();
  ctx.ellipse(150, 95, 60, 35, 0.2, 0, Math.PI * 2);
  ctx.ellipse(330, 145, 75, 45, -0.3, 0, Math.PI * 2);
  ctx.fill();

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

// Thin Inner Fresnel Rim GLSL Shader
const fresnelVertexShader = `
  varying vec3 vNormal;
  void main() {
    vNormal = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const fresnelFragmentShader = `
  varying vec3 vNormal;
  void main() {
    float f = pow(1.0 - max(dot(normalize(vNormal), vec3(0.0, 0.0, 1.0)), 0.0), 4.0);
    gl_FragColor = vec4(vec3(0.376, 0.773, 1.0), f * 0.55);
  }
`;

// Micro-satellite (small, realistic)
function createMicroSatellite() {
  const satellite = new THREE.Group();

  const bodyMaterial = new THREE.MeshStandardMaterial({
    color: 0x0f172a,
    metalness: 0.9,
    roughness: 0.25,
  });

  const goldMaterial = new THREE.MeshStandardMaterial({
    color: 0xd97706,
    metalness: 0.85,
    roughness: 0.3,
  });

  const panelMaterial = new THREE.MeshStandardMaterial({
    color: 0x0284c7,
    emissive: 0x0369a1,
    emissiveIntensity: 0.25,
    metalness: 0.7,
    roughness: 0.2,
  });

  const beaconMaterial = new THREE.MeshBasicMaterial({
    color: 0x38bdf8,
  });

  // Central bus
  const bus = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.07, 0.12), bodyMaterial);
  satellite.add(bus);

  // Gold telemetry section
  const goldUnit = new THREE.Mesh(new THREE.BoxGeometry(0.072, 0.072, 0.035), goldMaterial);
  goldUnit.position.z = 0.07;
  satellite.add(goldUnit);

  // Antenna dish
  const dish = new THREE.Mesh(new THREE.ConeGeometry(0.05, 0.03, 16, 1, true), bodyMaterial);
  dish.rotation.x = Math.PI / 2;
  dish.position.z = 0.1;
  satellite.add(dish);

  // Solar panels
  const leftPanel = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.08, 0.006), panelMaterial);
  leftPanel.position.set(-0.15, 0, 0);
  satellite.add(leftPanel);

  const rightPanel = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.08, 0.006), panelMaterial);
  rightPanel.position.set(0.15, 0, 0);
  satellite.add(rightPanel);

  // Glowing micro beacon
  const beacon = new THREE.Mesh(new THREE.SphereGeometry(0.012, 8, 8), beaconMaterial);
  beacon.position.set(0, 0.045, 0.06);
  satellite.add(beacon);

  const beaconLight = new THREE.PointLight(0x38bdf8, 0.35, 0.8);
  beaconLight.position.set(0, 0.045, 0.06);
  satellite.add(beaconLight);

  return satellite;
}

// Ultra-delicate orbit ring
function createOrbitRing(radiusX, radiusY, opacity = 0.14) {
  const segments = 128;
  const points = [];
  for (let i = 0; i <= segments; i++) {
    const angle = (i / segments) * Math.PI * 2;
    points.push(new THREE.Vector3(Math.cos(angle) * radiusX, Math.sin(angle) * radiusY, 0));
  }
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({
    color: 0x38bdf8,
    transparent: true,
    opacity,
  });
  return new THREE.Line(geometry, material);
}

export default function EarthScene() {
  const mountRef = useRef(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    let width = mount.clientWidth || window.innerWidth;
    let height = mount.clientHeight || window.innerHeight;

    // Three.js Scene & Camera setup
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(35, width / height, 0.1, 100);
    camera.position.set(0, 0, 7.5);

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: "high-performance",
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(width, height);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;

    mount.appendChild(renderer.domElement);

    // 1. Star Field (Cosmic Depth)
    const starCount = 1200;
    const starPositions = new Float32Array(starCount * 3);
    for (let i = 0; i < starCount; i++) {
      const radius = 16 + Math.random() * 22;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      starPositions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      starPositions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      starPositions[i * 3 + 2] = radius * Math.cos(phi);
    }
    const starGeometry = new THREE.BufferGeometry();
    starGeometry.setAttribute("position", new THREE.BufferAttribute(starPositions, 3));
    const starMaterial = new THREE.PointsMaterial({
      color: 0xe2e8f0,
      size: 0.026,
      transparent: true,
      opacity: 0.65,
      sizeAttenuation: true,
    });
    const stars = new THREE.Points(starGeometry, starMaterial);
    scene.add(stars);

    // 2. Realistic Cinematic Lighting
    // Soft HemisphereLight so dark side reveals subtle oceanic depth without pure black void
    const hemiLight = new THREE.HemisphereLight(0x93c5fd, 0x0b172a, 1.35);
    scene.add(hemiLight);

    // Key DirectionalLight from upper-front-left (-3.5, 2.5, 5.5)
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.1);
    keyLight.position.set(-3.5, 2.5, 5.5);
    scene.add(keyLight);

    // Subtle Cyan Rim Light from upper-right-back (4, 2, -3)
    const rimLight = new THREE.DirectionalLight(0x38bdf8, 1.2);
    rimLight.position.set(4, 2, -3);
    scene.add(rimLight);

    // 3. Earth System Setup (Occupies ~62% of viewport height)
    // Visible height at distance 7.5 and FOV 35 is ~4.73. 62% is ~2.94 diameter -> Radius = 1.47
    const EARTH_RADIUS = 1.47;
    const earthGroup = new THREE.Group();

    const textureLoader = new THREE.TextureLoader();
    textureLoader.setCrossOrigin("anonymous");

    // Textures with graceful fallback
    const earthAlbedo = textureLoader.load(
      EARTH_TEXTURE,
      (tex) => {
        tex.colorSpace = THREE.SRGBColorSpace;
      },
      undefined,
      () => {
        earthMesh.material.map = createProceduralEarthFallback();
        earthMesh.material.needsUpdate = true;
      }
    );
    earthAlbedo.colorSpace = THREE.SRGBColorSpace;

    const earthNormal = textureLoader.load(EARTH_NORMAL);
    const earthSpecular = textureLoader.load(EARTH_SPECULAR);

    // Earth Mesh - low shininess (6) and dark specular to prevent plastic blown-out highlights
    const earthGeometry = new THREE.SphereGeometry(EARTH_RADIUS, 64, 64);
    const earthMaterial = new THREE.MeshPhongMaterial({
      map: earthAlbedo,
      normalMap: earthNormal,
      specularMap: earthSpecular,
      specular: new THREE.Color(0x1e3a5f),
      shininess: 6,
    });
    const earthMesh = new THREE.Mesh(earthGeometry, earthMaterial);
    earthMesh.rotation.y = -1.1;
    earthGroup.add(earthMesh);

    // Cloud Layer
    const cloudTexture = textureLoader.load(EARTH_CLOUDS, (tex) => {
      tex.colorSpace = THREE.SRGBColorSpace;
    });
    cloudTexture.colorSpace = THREE.SRGBColorSpace;

    const cloudGeometry = new THREE.SphereGeometry(EARTH_RADIUS * 1.012, 64, 64);
    const cloudMaterial = new THREE.MeshPhongMaterial({
      map: cloudTexture,
      transparent: true,
      opacity: 0.28,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const cloudsMesh = new THREE.Mesh(cloudGeometry, cloudMaterial);
    cloudsMesh.rotation.y = -1.1;
    earthGroup.add(cloudsMesh);

    // 4. Atmosphere Layer 1: Soft Outer Atmospheric Halo (No Hard Edge)
    // THREE.Sprite positioned behind Earth, scaled ~2.85x Earth radius with smooth radial gradient
    const haloTexture = createAtmosphereHaloTexture();
    const haloMaterial = new THREE.SpriteMaterial({
      map: haloTexture,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const haloSprite = new THREE.Sprite(haloMaterial);
    haloSprite.scale.set(EARTH_RADIUS * 2.85 * 2, EARTH_RADIUS * 2.85 * 2, 1);
    haloSprite.position.set(0, 0, -0.06);
    earthGroup.add(haloSprite);

    // Atmosphere Layer 2: Thin Inner Fresnel Rim
    // Sphere radius * 1.015 with custom GLSL Fresnel shader tinted #60c5ff at 0.55 opacity
    const fresnelGeometry = new THREE.SphereGeometry(EARTH_RADIUS * 1.015, 64, 64);
    const fresnelMaterial = new THREE.ShaderMaterial({
      vertexShader: fresnelVertexShader,
      fragmentShader: fresnelFragmentShader,
      side: THREE.FrontSide,
      blending: THREE.AdditiveBlending,
      transparent: true,
      depthWrite: false,
    });
    const fresnelMesh = new THREE.Mesh(fresnelGeometry, fresnelMaterial);
    earthGroup.add(fresnelMesh);

    scene.add(earthGroup);

    // 5. Satellites & Delicate Orbits
    const orbitGroup = new THREE.Group();
    scene.add(orbitGroup);

    const orbitConfigs = [
      { rx: 2.25, ry: 1.65, rotX: 0.35, rotY: 0.15, rotZ: -0.22, speed: 0.14, angle: 0.8 },
      { rx: 2.55, ry: 1.95, rotX: 1.12, rotY: -0.25, rotZ: 0.4, speed: 0.10, angle: 2.6 },
      { rx: 2.38, ry: 1.78, rotX: -0.42, rotY: 0.55, rotZ: 0.18, speed: 0.12, angle: 4.5 },
      { rx: 2.65, ry: 2.1, rotX: 0.7, rotY: -0.35, rotZ: -0.55, speed: 0.085, angle: 1.5 },
    ];

    const satelliteEntries = [];

    orbitConfigs.forEach((cfg) => {
      const ringGroup = new THREE.Group();
      ringGroup.rotation.set(cfg.rotX, cfg.rotY, cfg.rotZ);

      const ring = createOrbitRing(cfg.rx, cfg.ry, 0.14);
      ringGroup.add(ring);

      const sat = createMicroSatellite();
      ringGroup.add(sat);

      orbitGroup.add(ringGroup);

      satelliteEntries.push({
        group: ringGroup,
        satellite: sat,
        rx: cfg.rx,
        ry: cfg.ry,
        speed: cfg.speed,
        angle: cfg.angle,
      });
    });

    // 6. Glowing 3D Particle Wave Across Lower Hero (85 x 45 grid)
    const waveCols = 85;
    const waveRows = 45;
    const waveCount = waveCols * waveRows;
    const wavePositions = new Float32Array(waveCount * 3);
    const waveColors = new Float32Array(waveCount * 3);

    const colCyan = new THREE.Color(0x38bdf8);
    const colBlue = new THREE.Color(0x2563eb);
    const colIce = new THREE.Color(0x93c5fd);

    let pIdx = 0;
    for (let r = 0; r < waveRows; r++) {
      const z = (r / (waveRows - 1)) * 4.8 + 1.0;
      for (let c = 0; c < waveCols; c++) {
        const x = (c / (waveCols - 1)) * 16.0 - 8.0;
        wavePositions[pIdx * 3] = x;
        wavePositions[pIdx * 3 + 1] = -1.65;
        wavePositions[pIdx * 3 + 2] = z;

        const factor1 = c / waveCols;
        const factor2 = r / waveRows;
        const col = colCyan.clone().lerp(colBlue, factor2 * 0.72).lerp(colIce, factor1 * 0.28);
        waveColors[pIdx * 3] = col.r;
        waveColors[pIdx * 3 + 1] = col.g;
        waveColors[pIdx * 3 + 2] = col.b;
        pIdx++;
      }
    }

    const waveGeometry = new THREE.BufferGeometry();
    waveGeometry.setAttribute("position", new THREE.BufferAttribute(wavePositions, 3));
    waveGeometry.setAttribute("color", new THREE.BufferAttribute(waveColors, 3));

    const waveDotTexture = createParticleDotTexture();
    const waveMaterial = new THREE.PointsMaterial({
      size: 0.075,
      map: waveDotTexture,
      vertexColors: true,
      transparent: true,
      opacity: 0.85,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const waveMesh = new THREE.Points(waveGeometry, waveMaterial);
    scene.add(waveMesh);

    // Responsive Positioning
    function updateLayout(w, h) {
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);

      const isMobile = w < 960;
      if (isMobile) {
        // Centered below headline on mobile
        earthGroup.position.set(0, -0.4, 0);
        earthGroup.scale.setScalar(0.75);
        orbitGroup.position.copy(earthGroup.position);
        orbitGroup.scale.setScalar(0.75);
      } else {
        // Desktop: sits cleanly at ~68-72% across width, vertically centered around y = 0.08
        // Visible width at z=0 is 4.73 * aspect. At 16:9, width is ~8.4, 70% from left is ~1.68
        const visibleHalfWidth = 2.365 * camera.aspect;
        const targetX = visibleHalfWidth * 0.42; // places center at ~71% across viewport
        earthGroup.position.set(targetX, 0.08, 0);
        earthGroup.scale.setScalar(1.0);
        orbitGroup.position.copy(earthGroup.position);
        orbitGroup.scale.setScalar(1.0);
      }
    }

    updateLayout(width, height);

    // 7. Interactive Physics, Cursor Tracking & Canvas Drag
    let mouseX = 0;
    let mouseY = 0;
    let isDragging = false;
    let prevPointerX = 0;
    let dragVelocityY = 0;
    let baseAngleY = 0;

    const handlePointerMove = (e) => {
      // Normalized window coordinates (-1 to 1)
      const x = (e.clientX / window.innerWidth) * 2 - 1;
      const y = -(e.clientY / window.innerHeight) * 2 + 1;
      mouseX = Math.max(-1, Math.min(1, x));
      mouseY = Math.max(-1, Math.min(1, y));

      if (isDragging) {
        const deltaX = e.clientX - prevPointerX;
        dragVelocityY += deltaX * 0.0035;
        prevPointerX = e.clientX;
      }
    };

    const handlePointerDown = (e) => {
      isDragging = true;
      prevPointerX = e.clientX;
    };

    const handlePointerUp = () => {
      isDragging = false;
    };

    window.addEventListener("pointermove", handlePointerMove, { passive: true });
    mount.addEventListener("pointerdown", handlePointerDown);
    window.addEventListener("pointerup", handlePointerUp);

    // 8. Animation Loop with clock
    const clock = new THREE.Clock();
    let animId;

    const animate = () => {
      animId = requestAnimationFrame(animate);
      const delta = Math.min(clock.getDelta(), 0.08);
      const elapsed = clock.getElapsedTime();

      // Inertia decay for manual spin
      dragVelocityY *= 0.94;

      // Cursor horizontal influence on rotation direction (left = anticlockwise, right = clockwise)
      const cursorDrift = mouseX * 0.14 * delta;
      const baseDrift = 0.045 * delta;
      baseAngleY += baseDrift + cursorDrift + dragVelocityY;

      earthMesh.rotation.y = baseAngleY;
      cloudsMesh.rotation.y = baseAngleY * 1.15;

      // Smooth pitch and yaw tilt with 0.045 damping
      const targetPitch = -mouseY * 0.22;
      const targetYaw = mouseX * 0.26;
      earthGroup.rotation.x += (targetPitch - earthGroup.rotation.x) * 0.045;
      earthGroup.rotation.z += (targetYaw * 0.32 - earthGroup.rotation.z) * 0.045;
      orbitGroup.rotation.x = earthGroup.rotation.x;
      orbitGroup.rotation.z = earthGroup.rotation.z;

      // Satellite progression
      satelliteEntries.forEach((entry) => {
        entry.angle += entry.speed * delta;
        const sx = Math.cos(entry.angle) * entry.rx;
        const sy = Math.sin(entry.angle) * entry.ry;
        entry.satellite.position.set(sx, sy, 0);
        entry.satellite.rotation.z = entry.angle + Math.PI / 2;
        entry.satellite.rotation.y += 0.01;
      });

      // Layered sine/cosine ripple wave beneath the Earth
      const pArr = waveGeometry.attributes.position.array;
      for (let i = 0; i < waveCount; i++) {
        const px = pArr[i * 3];
        const pz = pArr[i * 3 + 2];
        pArr[i * 3 + 1] =
          -1.65 +
          Math.sin(px * 0.42 + elapsed * 1.25) * 0.22 +
          Math.cos(pz * 0.65 + elapsed * 0.95) * 0.16 +
          Math.sin((px + pz) * 0.3 + elapsed * 0.7) * 0.08;
      }
      waveGeometry.attributes.position.needsUpdate = true;

      renderer.render(scene, camera);
    };

    animId = requestAnimationFrame(animate);

    // Resize listener
    const handleResize = () => {
      if (!mount) return;
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      if (w > 0 && h > 0) {
        updateLayout(w, h);
      }
    };
    window.addEventListener("resize", handleResize);

    // Clean Memory Disposal on Unmount
    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("pointermove", handlePointerMove);
      mount.removeEventListener("pointerdown", handlePointerDown);
      window.removeEventListener("pointerup", handlePointerUp);

      scene.traverse((obj) => {
        if (obj.geometry) {
          obj.geometry.dispose();
        }
        if (obj.material) {
          if (Array.isArray(obj.material)) {
            obj.material.forEach((m) => disposeMat(m));
          } else {
            disposeMat(obj.material);
          }
        }
      });

      function disposeMat(mat) {
        if (!mat) return;
        Object.keys(mat).forEach((k) => {
          const v = mat[k];
          if (v && typeof v.dispose === "function") {
            v.dispose();
          }
        });
        mat.dispose();
      }

      renderer.dispose();
      if (renderer.domElement && mount.contains(renderer.domElement)) {
        mount.removeChild(renderer.domElement);
      }
    };
  }, []);

  return <div ref={mountRef} className="earth-webgl" />;
}
