// Hero background: rows of grain plants (arching leaf blades, stem, drooping ear) swaying in the wind,
// drifting pollen and mouse parallax (like UnoffF1's hero). Wind is done in the vertex shader.
import * as THREE from "three";

const lerp = (a, b, t) => a + (b - a) * t;
const mix = (c1, c2, t) => c1.map((v, i) => lerp(v, c2[i], t));

// ---- plant geometry (1 unit ~ plant height) ----
function builder() {
  const pos = [], col = [], idx = [];
  // adds a rows x cols vertex grid as quads; f(i, j) -> [x, y, z, r, g, b]; wrap closes the columns into a tube
  const grid = (rows, cols, f, wrap = false) => {
    const o = pos.length / 3;
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) { const v = f(i, j); pos.push(v[0], v[1], v[2]); col.push(v[3], v[4], v[5]); }
    for (let i = 0; i < rows - 1; i++) for (let j = 0; j < (wrap ? cols : cols - 1); j++) {
      const a = o + i * cols + j, b = o + i * cols + ((j + 1) % cols), c = a + cols, d = b + cols;
      idx.push(a, c, b, b, c, d);
    }
  };
  return { grid, geometry() {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
    g.setIndex(idx); g.computeVertexNormals();
    return g;
  } };
}

// tube along a polyline of points with radius r(i); colour c(i)
function tube(b, pts, radius, colour, sides = 5) {
  b.grid(pts.length, sides, (i, j) => {
    const p = pts[i], q = pts[Math.min(i + 1, pts.length - 1)], pr = pts[Math.max(i - 1, 0)];
    const T = new THREE.Vector3(q[0] - pr[0], q[1] - pr[1], q[2] - pr[2]).normalize();
    const N1 = new THREE.Vector3().crossVectors(T, Math.abs(T.z) > 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 0, 1)).normalize();
    const N2 = new THREE.Vector3().crossVectors(T, N1);
    const a = (j / sides) * Math.PI * 2, r = radius(i / (pts.length - 1));
    return [p[0] + (N1.x * Math.cos(a) + N2.x * Math.sin(a)) * r, p[1] + (N1.y * Math.cos(a) + N2.y * Math.sin(a)) * r,
      p[2] + (N1.z * Math.cos(a) + N2.z * Math.sin(a)) * r, ...colour(i / (pts.length - 1))];
  }, true);
}

function plant(b, ox, oz, rnd) {
  const H = 0.95 + rnd() * 0.4, lean = (rnd() - 0.5) * 0.2, la = rnd() * 6.28;
  const green = [0.1 + rnd() * 0.04, 0.3 + rnd() * 0.1, 0.07];
  // stem
  const S = 8, stem = [];
  for (let i = 0; i <= S; i++) { const t = i / S; stem.push([ox + Math.cos(la) * lean * t * t, t * H, oz + Math.sin(la) * lean * t * t]); }
  tube(b, stem, (t) => 0.013 * (1 - 0.45 * t), (t) => mix([0.14, 0.3, 0.08], [0.45, 0.55, 0.18], t));
  // leaf blades: ribbons that rise then arch outward, tapering to a point
  const leaves = 4 + Math.floor(rnd() * 2);
  for (let k = 0; k < leaves; k++) {
    const a = (k / leaves) * 6.28 + rnd() * 0.9, y0 = 0.05 + k * 0.1 * H + rnd() * 0.1;
    const L = (0.55 + rnd() * 0.45) * H, reach = L * (0.35 + rnd() * 0.35), W = 0.05 + rnd() * 0.02;
    const dx = Math.cos(a), dz = Math.sin(a), wx = -dz, wz = dx, SEG = 9;
    b.grid(SEG + 1, 3, (i, j) => {
      const t = i / SEG, s = j - 1;
      const r = reach * t * t, y = y0 + L * (t - 0.42 * t * t);
      const w = W * Math.pow(1 - t, 0.65) * (0.3 + 0.7 * Math.min(1, t * 4)) * s;
      const lift = s === 0 ? 0.012 * (1 - t) : 0; // raised midrib = folded blade
      const c = mix(green, [0.55, 0.72, 0.22], Math.pow(t, 1.6)).map((v) => v * (s === 0 ? 1.3 : 1));
      return [ox + dx * r + wx * w, y + lift, oz + dz * r + wz * w, ...c];
    });
  }
  // ear: bumpy grain tube drooping from the stem top
  const E = 12, ear = [], top = stem[S], EL = 0.28 + rnd() * 0.08, th0 = 0.1;
  let p = [...top];
  for (let i = 0; i <= E; i++) {
    ear.push([...p]);
    const th = th0 + (i / E) * 1.25; // bends from vertical towards horizontal
    p = [p[0] + Math.sin(th) * Math.cos(la) * EL / E, p[1] + Math.cos(th) * EL / E, p[2] + Math.sin(th) * Math.sin(la) * EL / E];
  }
  const ripe = rnd();
  tube(b, ear, (u) => 0.034 * Math.pow(Math.sin(Math.PI * (0.12 + 0.88 * u)), 0.7) * (0.82 + 0.18 * Math.cos(u * 40)),
    (u) => mix(mix([0.45, 0.6, 0.2], [0.78, 0.62, 0.18], ripe), [0.9, 0.78, 0.35], u), 6);
}

function tuft(seed) {
  let s = seed; const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  const b = builder();
  for (let i = 0; i < 3; i++) plant(b, (rnd() - 0.5) * 0.28, (rnd() - 0.5) * 0.28, rnd);
  return b.geometry();
}

export function startHero(canvas, reduced) {
  let renderer;
  try { renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true }); } catch { return; } // no WebGL: CSS gradient remains
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x07120a, 0.085);
  const camera = new THREE.PerspectiveCamera(52, 1, 0.1, 80);
  camera.position.set(0, 1.0, 5.2);
  const field = new THREE.Group();
  scene.add(field);

  scene.add(new THREE.HemisphereLight(0xb8e89a, 0x0a1208, 1.1));
  const sun = new THREE.DirectionalLight(0xffd9a0, 2.4); // low back-left sun: rim-lights the blades and ears
  sun.position.set(-4, 2.2, -8);
  scene.add(sun);

  const ground = new THREE.Mesh(new THREE.PlaneGeometry(70, 70).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ color: 0x0b150b }));
  field.add(ground);

  // wind: bend grows with height squared, travelling as a wave across the field
  const uTime = { value: 0 };
  const mat = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide });
  mat.onBeforeCompile = (sh) => {
    sh.uniforms.uTime = uTime;
    sh.vertexShader = "uniform float uTime;\n" + sh.vertexShader.replace("#include <begin_vertex>", `#include <begin_vertex>
      float bend = position.y * position.y;
      vec4 wp = instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0);
      transformed.x += sin(uTime * 1.5 + wp.x * 0.45 + wp.z * 0.3) * 0.11 * bend + sin(uTime * 3.1 + wp.z) * 0.012 * bend;
      transformed.z += cos(uTime * 1.1 + wp.z * 0.35) * 0.05 * bend;`);
  };

  const ROWS = 40, PER = 62, variants = [tuft(7), tuft(91)];
  const count = Math.ceil((ROWS * PER) / 2);
  const meshes = variants.map((g) => new THREE.InstancedMesh(g, mat, count));
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), c = new THREE.Color(), n = [0, 0];
  for (let r = 0; r < ROWS; r++) for (let p = 0; p < PER; p++) {
    const v = (r + p) % 2, i = n[v]++, sc = 0.38 + Math.random() * 0.2; // plants ~0.45 units tall: rows recede like a real field
    q.setFromEuler(new THREE.Euler(0, Math.random() * 6.28, 0));
    m4.compose(new THREE.Vector3((r - ROWS / 2) * 0.55 + (Math.random() - 0.5) * 0.1, 0, 3 - p * 0.42 + (Math.random() - 0.5) * 0.1), q, new THREE.Vector3(sc, sc * (0.9 + Math.random() * 0.3), sc));
    meshes[v].setMatrixAt(i, m4);
    meshes[v].setColorAt(i, c.setScalar(0.75 + Math.random() * 0.4));
  }
  meshes.forEach((m, v) => { m.count = n[v]; m.frustumCulled = false; field.add(m); });

  // pollen
  const P = 220, pos = new Float32Array(P * 3);
  for (let i = 0; i < P; i++) pos.set([(Math.random() - 0.5) * 24, Math.random() * 3.5, 6 - Math.random() * 28], i * 3);
  const pg = new THREE.BufferGeometry();
  pg.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  scene.add(new THREE.Points(pg, new THREE.PointsMaterial({ size: 0.05, color: 0xd9ff8a, transparent: true, opacity: 0.7, depthWrite: false })));

  const resize = () => {
    const { clientWidth: w, clientHeight: h } = canvas;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(canvas);
  resize();

  let mx = 0, my = 0, visible = true;
  addEventListener("pointermove", (e) => { mx = e.clientX / innerWidth - 0.5; my = e.clientY / innerHeight - 0.5; });
  new IntersectionObserver(([e]) => (visible = e.isIntersecting)).observe(canvas);

  const frame = (t) => {
    uTime.value = reduced ? 0 : t / 1000;
    for (let i = 0; i < P; i++) { // drift up and sideways, wrap around
      pos[i * 3 + 1] += 0.004; pos[i * 3] += Math.sin(uTime.value + i) * 0.002;
      if (pos[i * 3 + 1] > 4) pos[i * 3 + 1] = 0;
    }
    pg.attributes.position.needsUpdate = true;
    field.rotation.y += (mx * 0.5 - field.rotation.y) * 0.05; // smoothed, like the F1 shirt viewer
    camera.position.y += (1.0 - my * 0.5 - camera.position.y) * 0.05;
    camera.lookAt(0, 0.35, -2);
    renderer.render(scene, camera);
  };

  if (reduced) return frame(0);
  renderer.setAnimationLoop((t) => { if (visible && !document.hidden) frame(t); });
}
