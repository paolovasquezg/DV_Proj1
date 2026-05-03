import * as d3 from "d3"

export const ratingColor = d3
  .scaleSequential()
  .domain([0, 10])
  .interpolator(
    d3.interpolateRgbBasis([
      "#79d7ef",
      "#e8f3f4",
      "#ffd6a6",
      "#ff714f",
      "#22151c"
    ])
  )

export function vsupColor(rating, cir) {
  if (!Number.isFinite(rating)) return "#e2e7e9"

  const r = Math.max(0, Math.min(10, rating))
  const c = Number.isFinite(cir) ? Math.max(0, Math.min(10, cir)) : 10
  const base = ratingColor(r)
  const uncertainty = Math.min(0.75, c / 10)

  return d3.interpolateLab(base, "#dfe5e7")(uncertainty)
}

export function statusFromAge(ageMinutes) {
  if (!Number.isFinite(ageMinutes)) return "missing"
  if (ageMinutes <= 15) return "fresh"
  if (ageMinutes <= 60) return "old"
  return "missing"
}

export const statusColor = {
  fresh: "#6bd9f2",
  old: "#9ca5aa",
  missing: "#e7ecef"
}