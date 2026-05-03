import * as XLSX from "xlsx"
import excelUrl from "../data/transf.xlsx?url"

function parse_date(value) {
  return new Date(value.replace(" ", "T"))
}

export async function load_excel() {
  const response = await fetch(excelUrl)
  const buffer = await response.arrayBuffer()
  const workbook = XLSX.read(buffer, { type: "array" })
  const sheet = workbook.Sheets[workbook.SheetNames[0]]
  const rows = XLSX.utils.sheet_to_json(sheet)

  return rows.map((d) => ({
    time: parse_date(d.time_bin), time_bin: d.time_bin, location: Number(d.location),
    category: d.category, map: Number(d.map), ci50_lo: Number(d.ci50_lo), ci50_hi: Number(d.ci50_hi),
    ci80_lo: Number(d.ci80_lo), ci80_hi: Number(d.ci80_hi), ci95_lo: Number(d.ci95_lo), ci95_hi: Number(d.ci95_hi),
    cir: Number(d.cir)
  }))
    .filter((d) => d.time && d.location && d.category)
    .sort((a, b) => +a.time - +b.time)
}