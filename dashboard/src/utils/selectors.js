import * as d3 from "d3"
import { LOCATIONS, ORDER } from "./constants"
import { GetStatus } from "./vsup"

export function GetTimes(data) {
  return Array.from(new Set(data.map((d) => +d.time))).sort(d3.ascending).map((d) => new Date(d))
}

export function GetCategories(data) {
  return Array.from(new Set(data.map((d) => d.category))).sort()
}

function missingReg(location, category) {
  return {
    location, category, time: null, map: null,
    ci50_lo: null, ci50_hi: null, ci80_lo: null, ci80_hi: null,
    ci95_lo: null, ci95_hi: null, cir: null, ageMinutes: Infinity, status: "missing"
  }
}

export function GetLatestRegs(data, category, selectedTime) {
  const cutoff = +selectedTime
  const byLocation = d3.group(data.filter((d) => d.category === category), (d) => d.location)

  const bisect = d3.bisector((d) => +d.time).right

  return LOCATIONS.map((location) => {
    const rows = byLocation.get(location) ?? []
    const latest = rows[bisect(rows, cutoff) - 1]

    if (!latest) return missingReg(location, category)

    const ageMinutes = (cutoff - +latest.time) / 60000
    return { ...latest, ageMinutes, status: GetStatus(ageMinutes) }
  })
}

export function GetAllCategoryRegs(data, selectedTime) {
  const cutoff = +selectedTime
  const bisect = d3.bisector((d) => +d.time).right

  return LOCATIONS.map((location) => {
    const locData = data.filter((d) => d.location === location)

    const catRegs = ORDER.map((cat) => {
      const rows = locData.filter((d) => d.category === cat)
      const latest = rows[bisect(rows, cutoff) - 1]
      if (!latest) return null
      const ageMinutes = (cutoff - +latest.time) / 60000
      return { ...latest, ageMinutes, status: GetStatus(ageMinutes) }
    }).filter(Boolean)

    if (!catRegs.length) return missingReg(location, "all")

    const maxReg = catRegs.reduce((best, r) => r.map > best.map ? r : best, catRegs[0])
    const minAge = d3.min(catRegs, (r) => r.ageMinutes)
    return { ...maxReg, category: "all", ageMinutes: minAge, status: GetStatus(minAge) }
  })
}