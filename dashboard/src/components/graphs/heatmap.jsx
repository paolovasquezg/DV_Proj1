import { useMemo } from "react"
import * as d3 from "d3"
import { LABELS, ORDER } from "../../utils/constants"
import { GetColorPalette } from "../../utils/vsup"

export default function HeatMap({ data, Location, Time, Category, setCategory }) {

  const width = 900
  const height = 185
  const margin = { top: 24, right: 18, bottom: 32, left: 125 }

  const Rows = useMemo(() => { return data.filter((d) => d.location === Location) }, [data, Location])

  const Categories = useMemo(() => {
    const existing = new Set(Rows.map((d) => d.category))

    return ORDER.filter((category) => existing.has(category))
  }, [Rows])

  const Cells = useMemo(() => {
    const best = new Map()
    for (const d of Rows) {
      const hour = +d3.timeHour.floor(d.time)
      const key = `${hour}|${d.category}`
      const prev = best.get(key)
      if (!prev || d.map > prev.map) best.set(key, { ...d, hour: new Date(hour) })
    }
    return Array.from(best.values())
  }, [Rows])

  const timeExtent = d3.extent(Rows, (d) => d.time)

  const x = d3.scaleTime().domain(timeExtent).range([margin.left, width - margin.right])
  const y = d3.scaleBand().domain(Categories).range([margin.top, height - margin.bottom]).padding(0.08)

  const hours = Array.from(new Set(Cells.map((d) => +d.hour))).sort(d3.ascending)
  const cellWidth = hours.length > 1 ? Math.max(2, x(new Date(hours[1])) - x(new Date(hours[0]))) : 6

  return (
    <svg className="block rounded-xl overflow-hidden" viewBox={`0 0 ${width} ${height}`}>
      <rect width={width} height={height} fill="white" />

      {Categories.map((category) => {
        const selected = category === Category

        return (
          <g key={category} onClick={() => setCategory(category)} style={{ cursor: "pointer" }}>

            <text x={margin.left - 10} y={y(category) + y.bandwidth() / 2 + 4} textAnchor="end" fontSize="10"
              fontWeight={selected ? "700" : "400"} fill={selected ? "#222" : "#555"}>
              {LABELS[category] ?? category}
            </text>

            {selected && (
              <rect x={margin.left - 4} y={y(category) - 2} width={width - margin.left - margin.right + 8}
                height={y.bandwidth() + 4} fill="none" stroke="#555" strokeWidth="1" opacity="0.35" />
            )}
          </g>)
      })}

      {Cells.map((d, index) => (
        <rect key={`${+d.hour}-${d.category}-${index}`} x={x(d.hour)} y={y(d.category)} width={cellWidth} height={y.bandwidth()} fill={GetColorPalette(d.map, d.cir, "vsup")}>
          <title>
            {`${LABELS[d.category] ?? d.category} ${d.hour.toLocaleString()} MAP: ${d.map?.toFixed?.(2)} CIR: ${d.cir?.toFixed?.(2)}`}
          </title>
        </rect>
      ))}

      {Time && (<line x1={x(Time)} x2={x(Time)} y1={margin.top - 6} y2={height - margin.bottom} stroke="#222" strokeWidth="1.5" />)}

      <g transform={`translate(0, ${height - margin.bottom})`}>
        {x.ticks(7).map((tick) => (

          <g key={+tick} transform={`translate(${x(tick)},0)`}>
            <line y2="5" stroke="#999" />
            <text y="18" textAnchor="middle" fontSize="9" fill="#555">
              {d3.timeFormat("%a %d %H:%M")(tick)}
            </text>
          </g>

        ))}
        <line x1={margin.left} x2={width - margin.right} stroke="#ccc" />
      </g>
    </svg>
  )
}