import * as d3 from "d3"

export const RatingColor = d3.scaleSequential().domain([0, 10]).interpolator(
  d3.interpolateRgbBasis(["#7ce2ff", "#f8e19d", "#ffb469", "#ff813d", "#ff3c10", "#c20d06", "#710c06", "#2d0300"]))

export function GetColorPalette(map, cir, palette) {
  if (map === null || map === undefined) return "#e7ecef"
  if (palette === "normal") return RatingColor(map)
  const rating = RatingColor(map)
  if (palette === "vsup_ext") {
    return d3.interpolateLab(rating, "#f2ab88ff")(Math.min((cir ?? 0) / 7, 1))
  }
  return d3.interpolateLab(rating, "#ffd9c6")((cir ?? 0) / 10)
}

export function GetStatus(ageMinutes) {
  if (ageMinutes <= 15) return "fresh"
  if (ageMinutes <= 60) return "old"
  return "missing"
}

export const StatusColor = { fresh: "#6bd9f2", old: "#9ca5aa", missing: "#e7ecef" }