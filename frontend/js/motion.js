// GSAP + Lenis motion layer (same stack as UnoffF1). Globals come from vendor/*.min.js.
const { gsap, ScrollTrigger, SplitText, Lenis } = window;
gsap.registerPlugin(ScrollTrigger, SplitText);
gsap.defaults({ ease: "expo.out" });

export const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

export function initScroll() {
  if (reduced) return;
  const lenis = new Lenis({ anchors: true });
  lenis.on("scroll", ScrollTrigger.update);
  gsap.ticker.add((t) => lenis.raf(t * 1000));
  gsap.ticker.lagSmoothing(0);
}

const fmt = (v, el) => {
  const d = +(el.dataset.decimals || 0);
  const s = el.dataset.sep ? Math.round(v).toLocaleString("en-US") : v.toFixed(d);
  return s + (el.dataset.suffix || "");
};

// Counts a number up from 0 to its data-count (or an explicit target).
export function countUp(el, to = +el.dataset.count, duration = 1.6) {
  if (reduced) return (el.textContent = fmt(to, el));
  const o = { v: 0 };
  gsap.to(o, { v: to, duration, onUpdate: () => (el.textContent = fmt(o.v, el)) });
}

export function intro() {
  const stats = document.querySelectorAll("#hero-stats dd");
  if (reduced) return stats.forEach((el) => countUp(el));
  const split = SplitText.create("#hero-title", { type: "lines,chars", mask: "lines", linesClass: "line" });
  gsap.timeline({ delay: 0.15 })
    .from(".eyebrow", { opacity: 0, y: 12, duration: 0.8 })
    .from(split.chars, { yPercent: 110, rotate: 6, duration: 1.1, stagger: { amount: 0.5 } }, 0.1)
    .from(["#hero-lede", "#hero-cta"], { opacity: 0, y: 24, duration: 1, stagger: 0.12 }, 0.7)
    .from("#hero-stats > div", { opacity: 0, y: 30, duration: 1, stagger: 0.1 }, 0.9)
    .add(() => stats.forEach((el) => countUp(el)), 1.0);
  gsap.to(".hero-copy", { yPercent: -12, opacity: 0.2, ease: "none",
    scrollTrigger: { trigger: ".hero", start: "top top", end: "bottom top", scrub: true } });
}

export function scrollReveals() {
  if (reduced) return;
  gsap.utils.toArray("[data-reveal]").forEach((el, i) =>
    gsap.from(el, { opacity: 0, y: 40, duration: 1.1, delay: (i % 4) * 0.06,
      scrollTrigger: { trigger: el, start: "top 88%", once: true } }));
}

// Called after every render: cards slide in, bars fill, percentages and the radar draw themselves.
export function animateResults(root) {
  const cards = root.querySelectorAll(".card");
  const bars = root.querySelectorAll(".bar i");
  const nums = root.querySelectorAll("[data-count]");
  if (reduced) return nums.forEach((el) => countUp(el));
  gsap.from(cards, { opacity: 0, y: 36, duration: 0.9, stagger: 0.09, clearProps: "opacity,transform" });
  bars.forEach((b) => gsap.fromTo(b, { "--w": "0%" }, { "--w": b.dataset.w, duration: 1.2, delay: 0.25 }));
  nums.forEach((el) => countUp(el, +el.dataset.count, 1.2));
  root.querySelectorAll(".radar-poly").forEach((p) =>
    gsap.from(p, { scale: 0, transformOrigin: "50% 50%", duration: 1.1, ease: "back.out(1.4)", delay: 0.2 }));
}

export function pulseCrop(el) {
  if (reduced) return;
  const s = SplitText.create(el, { type: "chars" });
  gsap.from(s.chars, { yPercent: 80, opacity: 0, duration: 0.8, stagger: 0.03 });
}
