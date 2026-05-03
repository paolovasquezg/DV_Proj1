import { useMemo } from "react"
import * as d3 from "d3"
import { NEIGHBOURHOOD_NAMES } from "../utils/constants"
import { statusColor, vsupColor } from "../utils/vsup"

export default function ErrorBarChart({ rows, selectedLocation, sortMode }) {
  const width = 650
  const height = 470
  const margin = { top: 58, right: 28, bottom: 28, left: 70 }

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
    <svg className="panel" viewBox={`0 0 ${width} ${height}`}>
      <rect width={width} height={height} fill="#fbfbfb" />

      <text x={margin.left} y={18} fontSize="12" fontWeight="700">
        Rating
      </text>

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

      <text x={14} y={margin.top - 8} fontSize="11" fontWeight="700" transform={`rotate(-90 14 ${margin.top - 8})`}>
        Neighbourhood
      </text>

      {sortedRows.map((d) => {
        const cy = y(d.location) + y.bandwidth() / 2
        const selected = d.location === selectedLocation

        return (
          <g key={d.location}>
            <text x={margin.left - 12} y={cy + 4} textAnchor="end" fontSize="10" fill="#444">
              {d.location}
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

            {selected && (
              <text x={x(d.map) + 12} y={cy + 4} fontSize="11" fontWeight="700" fill="#555">
                {NEIGHBOURHOOD_NAMES[d.location]}
              </text>
            )}
          </g>
        )
      })}

      <g transform={`translate(${width - 95}, 60)`}>
        <text x="0" y="0" fontSize="11" fontWeight="700">
          95% CIR
        </text>
        <text x="0" y="18" fontSize="10" fill="#555">
          0 - 1.25
        </text>
        <text x="0" y="34" fontSize="10" fill="#555">
          1.25 - 2.50
        </text>
        <text x="0" y="50" fontSize="10" fill="#555">
          2.50 - 5.00
        </text>
        <text x="0" y="66" fontSize="10" fill="#555">
          &gt; 5.00
        </text>
      </g>
    </svg>
  )
}