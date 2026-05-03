import { useMemo } from "react"
import * as d3 from "d3"
import { NEIGHBOURHOOD_NAMES } from "../utils/constants"
import { statusColor, vsupColor } from "../utils/vsup"

export default function ErrorBarChart({ rows, selectedLocation, setSelectedLocation, sortMode }) {
  const width = 650
  const height = 470
  const margin = { top: 58, right: 28, bottom: 28, left: 110 }

  const sortedRows = useMemo(() => {
    const clean = rows.filter((d) => Number.isFinite(d.map))

    return clean.slice().sort((a, b) => {
      if (sortMode === "map") return d3.descending(a.map, b.map)
      if (sortMode === "cir") return d3.ascending(a.cir, b.cir)
      if (sortMode === "location") return d3.ascending(a.location, b.location)
      return d3.descending(a.ci95_lo, b.ci95_lo)
    })
  }, [rows, sortMode])

  const x = d3.scaleLinear().domain([0, 10]).range([margin.left, width - margin.right])
  const y = d3.scaleBand().domain(sortedRows.map((d) => d.location)).range([margin.top, height - margin.bottom]).padding(0.32)

  return (
    <svg className="w-full block rounded-xl overflow-hidden" viewBox={`0 0 ${width} ${height}`}>
      <rect width={width} height={height} fill="white" />

      {d3.range(0, 10, 0.25).map((v) => (
        <rect
          key={v}
          x={x(v)}
          y={29}
          width={Math.max(1, x(v + 0.25) - x(v))}
          height={16}
          fill={vsupColor(v, 0.5)}
        />
      ))}

      {x.ticks(10).map((tick) => (
        <g key={tick}>
          <line x1={x(tick)} x2={x(tick)} y1={margin.top} y2={height - margin.bottom} stroke="#e7e7e7" />
          <text x={x(tick)} y={25} textAnchor="middle" fontSize="10" fill="#333">
            {tick}
          </text>
        </g>
      ))}


      {sortedRows.map((d) => {
        const cy = y(d.location) + y.bandwidth() / 2
        const selected = d.location === selectedLocation

        return (
          <g key={d.location} onClick={() => setSelectedLocation(d.location)} style={{ cursor: "pointer" }}>
            <text x={margin.left - 12} y={cy + 4} textAnchor="end" fontSize="10" fill={selected ? "#222" : "#444"} fontWeight={selected ? "700" : "400"}>
              {NEIGHBOURHOOD_NAMES[d.location] ?? d.location}
            </text>

            <line
              x1={x(Math.max(0, d.ci95_lo))}
              x2={x(Math.min(10, d.ci95_hi))}
              y1={cy}
              y2={cy}
              stroke="#d0d5d8"
              strokeWidth="6"
              strokeLinecap="round"
            />

            <line
              x1={x(Math.max(0, d.ci80_lo))}
              x2={x(Math.min(10, d.ci80_hi))}
              y1={cy}
              y2={cy}
              stroke="#aeb8bd"
              strokeWidth="4"
              strokeLinecap="round"
            />

            <line
              x1={x(Math.max(0, d.ci50_lo))}
              x2={x(Math.min(10, d.ci50_hi))}
              y1={cy}
              y2={cy}
              stroke="#7f8b91"
              strokeWidth="2"
              strokeLinecap="round"
            />

            <circle
              cx={x(d.map)}
              cy={cy}
              r={selected ? 7 : 5}
              fill={selected ? "#f5a77c" : vsupColor(d.map, d.cir)}
              stroke={statusColor[d.status]}
              strokeWidth="3"
            />

          </g>
        )
      })}
    </svg>
  )
}