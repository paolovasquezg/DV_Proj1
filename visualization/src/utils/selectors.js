import * as d3 from "d3"
import { LOCATIONS } from "./constants"
import { statusFromAge } from "./vsup"

export function getTimes(data) {
  return Array.from(new Set(data.map((d) => +d.time)))
    .sort(d3.ascending)
    .map((d) => new Date(d))
}

export function getCategories(data) {
  return Array.from(new Set(data.map((d) => d.category))).sort()
}

export function latestRowsFor(data, category, selectedTime) {
  const t = +selectedTime
  const grouped = d3.group(
    data.filter((d) => d.category === category),
    (d) => d.location
  )

  const bisect = d3.bisector((d) => +d.time).right

  return LOCATIONS.map((location) => {
    const rows = grouped.get(location) ?? []
    const i = bisect(rows, t) - 1
    const row = rows[i]

    if (!row) {
      return {
        location,
        category,
        time: null,
        map: null,
        ci50_lo: null,
        ci50_hi: null,
        ci80_lo: null,
        ci80_hi: null,
        ci95_lo: null,
        ci95_hi: null,
        cir: null,
        ageMinutes: Infinity,
        status: "missing"
      }
    }

    const ageMinutes = (t - +row.time) / 60000

    return {
      ...row,
      ageMinutes,
      status: statusFromAge(ageMinutes)
    }
  })
}