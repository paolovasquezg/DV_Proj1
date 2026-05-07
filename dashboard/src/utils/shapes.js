import coords from "../data/counties.geo.json"
import { scaleLinear } from "d3-scale"
import { extent } from "d3-array"

const WIDTH = 550
const HEIGHT = 400

const all_coords = coords.features.flatMap((f) => f.geometry.coordinates[0])

const xScale = scaleLinear().domain(extent(all_coords, (p) => p[0])).range([0, WIDTH])

const yScale = scaleLinear().domain(extent(all_coords, (p) => p[1])).range([HEIGHT, 0])

function normalize([x, y]) { return [xScale(x), yScale(y)] }

export const CountyShapes = coords.features.map((f) => {

  const points = f.geometry.coordinates[0].map(normalize)

  const cx = points.reduce((a, p) => a + p[0], 0) / points.length

  const cy = points.reduce((a, p) => a + p[1], 0) / points.length

  return { id: f.properties.loc, name: f.properties.locName, c: [cx, cy], p: points }
})

export function CountyForm(points) {
  return `M${points.map((p) => p.join(",")).join("L")}Z`
}