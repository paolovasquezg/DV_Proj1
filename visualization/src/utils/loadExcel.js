import * as XLSX from "xlsx"
import excelUrl from "../data/mc1_bsts_map_cir_py.xlsx?url"

function parseDate(value) {
  if (value instanceof Date) return value

  if (typeof value === "number") {
    const parsed = XLSX.SSF.parse_date_code(value)
    if (!parsed) return null
    return new Date(parsed.y, parsed.m - 1, parsed.d, parsed.H, parsed.M, parsed.S)
  }

  if (typeof value === "string") {
    const date = new Date(value.replace(" ", "T"))
    return Number.isNaN(+date) ? null : date
  }

  return null
}

function number(value) {
  const n = Number(value)
  return Number.isFinite(n) ? n : null
}

export async function loadMc1Excel() {
  const response = await fetch(excelUrl)
  const buffer = await response.arrayBuffer()
  const workbook = XLSX.read(buffer, { type: "array" })
  const sheet = workbook.Sheets[workbook.SheetNames[0]]
  const rows = XLSX.utils.sheet_to_json(sheet)

  return rows
    .map((d) => ({
      time: parseDate(d.time_bin ?? d.time ?? d.timestamp),
      time_bin: d.time_bin ?? d.time ?? d.timestamp,
      location: number(d.location),
      category: d.category,
      map: number(d.map ?? d.MAP),
      ci50_lo: number(d.ci50_lo),
      ci50_hi: number(d.ci50_hi),
      ci80_lo: number(d.ci80_lo),
      ci80_hi: number(d.ci80_hi),
      ci95_lo: number(d.ci95_lo),
      ci95_hi: number(d.ci95_hi),
      cir: number(d.cir ?? d.CIR)
    }))
    .filter((d) => d.time && d.location && d.category)
    .sort((a, b) => +a.time - +b.time)
}