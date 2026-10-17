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

function createSatellite() {
  const satellite = new THREE.Group();

  const bodyMaterial = new THREE.MeshStandardMaterial({
    color: 0x181923,
    metalness: 0.85,
    roughness: 0.3,
  });

  const darkMaterial = new THREE.MeshStandardMaterial({
    color: 0x08090f,
    metalness: 0.9,
    roughness: 0.22,
  });

  const silverMaterial = new THREE.MeshStandardMaterial({
    color: 0x858694,
    metalness: 0.9,
    roughness: 0.22,
  });

  const panelMaterial = new THREE.MeshStandardMaterial({
    color: 0x142d58,
    metalness: 0.55,
    roughness: 0.35,
  });

  const panelLineMaterial = new THREE.LineBasicMaterial({
    color: 0x6076a5,
    transparent: true,
    opacity: 0.72,
  });

  const body = new THREE.Mesh(
    new THREE.CylinderGeometry(0.23, 0.27, 0.72, 16),
    bodyMaterial
  );
  body.rotation.z = Math.PI / 2;
  satellite.add(body);

  const front = new THREE.Mesh(
    new THREE.CylinderGeometry(0.34, 0.34, 0.1, 20),
    silverMaterial
  );
  front.rotation.z = Math.PI / 2;
  front.position.x = 0.38;
  satellite.add(front);

  const rear = new THREE.Mesh(
    new THREE.CylinderGeometry(0.25, 0.29, 0.16, 16),
    darkMaterial
  );
  rear.rotation.z = Math.PI / 2;
  rear.position.x = -0.41;
  satellite.add(rear);

  const dish = new THREE.Mesh(
    new THREE.ConeGeometry(0.42, 0.18, 32, 1, true),
    silverMaterial
  );
  dish.rotation.z = -Math.PI / 2;
  dish.position.x = 0.54;
  satellite.add(dish);

  const dishCenter = new THREE.Mesh(
    new THREE.CylinderGeometry(0.035, 0.035, 0.28, 10),
    darkMaterial
  );
  dishCenter.rotation.z = Math.PI / 2;
  dishCenter.position.x = 0.68;
  satellite.add(dishCenter);

  const boom = new THREE.Mesh(
    new THREE.CylinderGeometry(0.025, 0.025, 1.45, 8),
    silverMaterial
  );
  boom.rotation.z = Math.PI / 2;
  satellite.add(boom);

  function createSolarPanel(side) {
    const panelGroup = new THREE.Group();
    const panel = new THREE.Mesh(
      new THREE.BoxGeometry(0.82, 0.045, 0.62),
      panelMaterial
    );
    panelGroup.add(panel);

    const frame = new THREE.LineSegments(
      new THREE.EdgesGeometry(panel.geometry),
      panelLineMaterial
    );
    frame.scale.set(1.02, 1.02, 1.02);
    panelGroup.add(frame);

    for (let x = -0.28; x <= 0.28; x += 0.28) {
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(x, 0.025, -0.3),
          new THREE.Vector3(x, 0.025, 0.3),
        ]),
        panelLineMaterial
      );
      panelGroup.add(line);
    }

    for (let z = -0.2; z <= 0.2; z += 0.2) {
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(-0.41, 0.025, z),
          new THREE.Vector3(0.41, 0.025, z),
        ]),
        panelLineMaterial
      );
      panelGroup.add(line);
    }

    panelGroup.position.x = side * 0.88;
    panelGroup.rotation.z = side * 0.015;
    satellite.add(panelGroup);
  }

  createSolarPanel(-1);
  createSolarPanel(1);

  const pinkLight = new THREE.PointLight(0xff2da6, 1.2, 2.2);
  pinkLight.position.set(0.48, 0.18, 0.25);
  satellite.add(pinkLight);

  satellite.scale.setScalar(0.72);
  return satellite;
}

function createOrbit(radiusX, radiusY, rotationZ, opacity = 0.58) {
  const points = [];
  for (let i = 0; i <= 160; i++) {
    const angle = (i / 160) * Math.PI * 2;
    points.push(
      new THREE.Vector3(
        Math.cos(angle) * radiusX,
        Math.sin(angle) * radiusY,
        0
      )
    );
  }

  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({
    color: 0xff27ad,
    transparent: true,
    opacity,
  });

  const orbit = new THREE.Line(geometry, material);
  orbit.rotation.z = rotationZ;
  return orbit;
}

function createStars(scene) {
  const geometry = new THREE.BufferGeometry();
  const count = 1800;
  const positions = new Float32Array(count * 3);

  for (let i = 0; i < count; i++) {
    const radius = 12 + Math.random() * 18;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);

    positions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
    positions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
    positions[i * 3 + 2] = radius * Math.cos(phi);
  }

  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));

  const material = new THREE.PointsMaterial({
    color: 0xffffff,
    size: 0.035,
    transparent: true,
    opacity: 0.8,
    sizeAttenuation: true,
  });

  scene.add(new THREE.Points(geometry, material));
}

export default function EarthScene() {
  const mountRef = useRef(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const width = mount.clientWidth || window.innerWidth;
    const height = mount.clientHeight || window.innerHeight;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(34, width / height, 0.1, 100);
    camera.position.set(0, 0.15, 7.3);

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: "high-performance",
    });

    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;

    mount.appendChild(renderer.domElement);

    createStars(scene);

    const ambient = new THREE.AmbientLight(0x25345e, 1.1);
    scene.add(ambient);

    const sun = new THREE.DirectionalLight(0xffffff, 3.3);
    sun.position.set(-4, 3, 6);
    scene.add(sun);

    const pinkFill = new THREE.PointLight(0xff1ba9, 5, 12);
    pinkFill.position.set(4, 1, 3);
    scene.add(pinkFill);

    const earthSystem = new THREE.Group();
    const earthGeometry = new THREE.SphereGeometry(2.18, 64, 64);
    const textureLoader = new THREE.TextureLoader();

    const earthTexture = textureLoader.load(EARTH_TEXTURE);
    earthTexture.colorSpace = THREE.SRGBColorSpace;
    const normalTexture = textureLoader.load(EARTH_NORMAL);
    const specularTexture = textureLoader.load(EARTH_SPECULAR);

    const earthMaterial = new THREE.MeshPhongMaterial({
      map: earthTexture,
      normalMap: normalTexture,
      specularMap: specularTexture,
      specular: new THREE.Color(0x557799),
      shininess: 12,
    });

    const earth = new THREE.Mesh(earthGeometry, earthMaterial);
    earth.rotation.y = -0.8;
    earthSystem.add(earth);

    const cloudTexture = textureLoader.load(EARTH_CLOUDS);
    cloudTexture.colorSpace = THREE.SRGBColorSpace;

    const cloudMaterial = new THREE.MeshPhongMaterial({
      map: cloudTexture,
      transparent: true,
      opacity: 0.34,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    const clouds = new THREE.Mesh(
      new THREE.SphereGeometry(2.205, 64, 64),
      cloudMaterial
    );
    clouds.rotation.y = -0.8;
    earthSystem.add(clouds);

    const atmosphereMaterial = new THREE.MeshBasicMaterial({
      color: 0x2186ff,
      transparent: true,
      opacity: 0.15,
      side: THREE.BackSide,
      blending: THREE.AdditiveBlending,
    });

    const atmosphere = new THREE.Mesh(
      new THREE.SphereGeometry(2.29, 64, 64),
      atmosphereMaterial
    );
    earthSystem.add(atmosphere);

    const atmosphereGlowMaterial = new THREE.MeshBasicMaterial({
      color: 0x0c4dff,
      transparent: true,
      opacity: 0.055,
      side: THREE.BackSide,
      blending: THREE.AdditiveBlending,
    });

    const atmosphereGlow = new THREE.Mesh(
      new THREE.SphereGeometry(2.42, 48, 48),
      atmosphereGlowMaterial
    );
    earthSystem.add(atmosphereGlow);

    earthSystem.position.set(1.05, -0.1, 0);
    earthSystem.scale.setScalar(0.96);
    scene.add(earthSystem);

    const orbitSystem = new THREE.Group();
    const orbitOne = createOrbit(3.35, 1.05, -0.18, 0.7);
    const orbitTwo = createOrbit(3.65, 1.35, 0.28, 0.58);
    const orbitThree = createOrbit(3.55, 1.62, -0.43, 0.48);

    orbitSystem.add(orbitOne);
    orbitSystem.add(orbitTwo);
    orbitSystem.add(orbitThree);
    orbitSystem.position.copy(earthSystem.position);
    scene.add(orbitSystem);

    const satellites = [];

    const satelliteOne = createSatellite();
    satelliteOne.position.set(-2.1, 1.15, 0.45);
    satelliteOne.rotation.set(0.25, -0.4, -0.15);
    satelliteOne.scale.multiplyScalar(1.08);
    orbitSystem.add(satelliteOne);
    satellites.push({
      object: satelliteOne,
      radiusX: 3.35,
      radiusY: 1.05,
      angle: 2.52,
      speed: 0.11,
    });

    const satelliteTwo = createSatellite();
    satelliteTwo.position.set(3.0, 1.2, -0.1);
    satelliteTwo.rotation.set(-0.25, 0.45, 0.2);
    satelliteTwo.scale.multiplyScalar(0.86);
    orbitSystem.add(satelliteTwo);
    satellites.push({
      object: satelliteTwo,
      radiusX: 3.65,
      radiusY: 1.35,
      angle: 0.55,
      speed: 0.075,
    });

    const satelliteThree = createSatellite();
    satelliteThree.position.set(2.6, -1.5, 0.3);
    satelliteThree.rotation.set(0.3, 0.15, -0.2);
    satelliteThree.scale.multiplyScalar(0.82);
    orbitSystem.add(satelliteThree);
    satellites.push({
      object: satelliteThree,
      radiusX: 3.55,
      radiusY: 1.62,
      angle: -0.55,
      speed: 0.09,
    });

    const satelliteFour = createSatellite();
    satelliteFour.position.set(-2.8, -1.05, -0.3);
    satelliteFour.rotation.set(-0.2, -0.2, 0.25);
    satelliteFour.scale.multiplyScalar(0.68);
    orbitSystem.add(satelliteFour);
    satellites.push({
      object: satelliteFour,
      radiusX: 3.35,
      radiusY: 1.05,
      angle: 3.75,
      speed: 0.065,
    });

    const clock = new THREE.Clock();

    function animate() {
      const elapsed = clock.getElapsedTime();
      earth.rotation.y += 0.0017;
      clouds.rotation.y += 0.0022;

      orbitSystem.rotation.z = Math.sin(elapsed * 0.04) * 0.025;

      satellites.forEach((item, index) => {
        item.angle += item.speed * 0.01;
        const x = Math.cos(item.angle) * item.radiusX;
        const y = Math.sin(item.angle) * item.radiusY;

        item.object.position.set(
          x,
          y,
          Math.sin(item.angle * 1.7 + index) * 0.55
        );
        item.object.rotation.y += 0.004;
        item.object.rotation.x += 0.001;
      });

      renderer.render(scene, camera);
    }

    renderer.setAnimationLoop(animate);

    function resize() {
      if (!mount) return;
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      if (w === 0 || h === 0) return;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    }

    window.addEventListener("resize", resize);

    return () => {
      window.removeEventListener("resize", resize);
      renderer.setAnimationLoop(null);
      renderer.dispose();

      earthGeometry.dispose();
      earthMaterial.dispose();
      cloudMaterial.dispose();
      atmosphereMaterial.dispose();
      atmosphereGlowMaterial.dispose();

      if (renderer.domElement && mount.contains(renderer.domElement)) {
        mount.removeChild(renderer.domElement);
      }
    };
  }, []);

  return <div ref={mountRef} className="earth-webgl" />;
}
