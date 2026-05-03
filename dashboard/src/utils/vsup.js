import * as d3 from "d3"

export const RatingColor = d3.scaleSequential().domain([0, 10]).interpolator(
  d3.interpolateRgbBasis(["#79d7ef", "#e8f3f4", "#ffd6a6", "#ff714f", "#22151c"]))

export function GetColor(map, cir) {

  const rating = RatingColor(map)
  const uncertainty = cir / 10

  return d3.interpolateLab(rating, "#c8c8c8")(uncertainty)
}

export function GetStatus(ageMinutes) {
  if (ageMinutes <= 15) return "fresh"
  if (ageMinutes <= 60) return "old"
  return "missing"
}

export const StatusColor = { fresh: "#6bd9f2", old: "#9ca5aa", missing: "#e7ecef" }