import { useMemo } from "react"
import { LABELS, ORDER } from "../../utils/constants"
import { GetColor } from "../../utils/vsup"
import * as d3 from "d3"

export default function Lines({ data, Location, Time, Category, setCategory }) {
  const width = 900
  const rowHeight = 92
  const margin = { top: 16, right: 38, bottom: 32, left: 125 }

  const Rating = (value) => {
    if (!Number.isFinite(value)) return 0
    return Math.max(0, Math.min(10, value))
  }

  const simulatedDots = (rows) => {
    const offsets = [-0.45, -0.15, 0.2, 0.42]
    return rows
      .filter((_, i) => i % 5 === 0)
      .flatMap((row, i) =>
        offsets.map((offset, j) => {
          const spread = Math.max(0.15, Math.min(1.2, (row.cir ?? 1) / 5))
          return { id: `${+row.time}-${i}-${j}`, time: row.time, value: Rating(row.map + offset * spread), map: row.map, cir: row.cir }
        })
      )
  }

  const Rows = useMemo(() => data.filter((d) => d.location === Location), [data, Location])

  const Categories = useMemo(() => {
    const existing = new Set(Rows.map((d) => d.category))
    return ORDER.filter((cat) => existing.has(cat))
  }, [Rows])

  const Grouped = useMemo(() => d3.group(Rows, (d) => d.category), [Rows])

  const height = margin.top + margin.bottom + Categories.length * rowHeight
  const timeExtent = d3.extent(Rows, (d) => d.time)

  const x = d3.scaleTime().domain(timeExtent).range([margin.left, width - margin.right])
  const y = d3.scaleLinear().domain([0, 10]).range([rowHeight - 18, 14])

  const line = d3.line().defined((d) => Number.isFinite(d.map)).x((d) => x(d.time)).y((d) => y(Rating(d.map)))
  const area95 = d3.area().defined((d) => Number.isFinite(d.ci95_lo) && Number.isFinite(d.ci95_hi)).x((d) => x(d.time)).y0((d) => y(Rating(d.ci95_lo))).y1((d) => y(Rating(d.ci95_hi)))
  const area80 = d3.area().defined((d) => Number.isFinite(d.ci80_lo) && Number.isFinite(d.ci80_hi)).x((d) => x(d.time)).y0((d) => y(Rating(d.ci80_lo))).y1((d) => y(Rating(d.ci80_hi)))
  const area50 = d3.area().defined((d) => Number.isFinite(d.ci50_lo) && Number.isFinite(d.ci50_hi)).x((d) => x(d.time)).y0((d) => y(Rating(d.ci50_lo))).y1((d) => y(Rating(d.ci50_hi)))

  return (
    <svg className="w-full block rounded-xl overflow-hidden" viewBox={`0 0 ${width} ${height}`}>
      <rect width={width} height={height} fill="white" />

      {Categories.map((category, index) => {
        const rows = Grouped.get(category) ?? []
        const selected = category === Category
        const offsetY = margin.top + index * rowHeight

        return (
          <g key={category} transform={`translate(0,${offsetY})`} onClick={() => setCategory(category)} style={{ cursor: "pointer" }}>

            <text x={margin.left - 10} y={rowHeight / 2} textAnchor="end" fontSize="10" fontWeight={selected ? "700" : "400"} fill={selected ? "#222" : "#555"}>
              {LABELS[category] ?? category}
            </text>

            <line x1={margin.left} x2={width - margin.right} y1={y(0)} y2={y(0)} stroke="#ddd" />
            <line x1={margin.left} x2={width - margin.right} y1={y(5)} y2={y(5)} stroke="#e5e5e5" />
            <line x1={margin.left} x2={width - margin.right} y1={y(10)} y2={y(10)} stroke="#ddd" />

            <path d={area95(rows)} fill="#d9dddf" opacity="0.75" />
            <path d={area80(rows)} fill="#bdc5c9" opacity="0.65" />
            <path d={area50(rows)} fill="#9aa5aa" opacity="0.55" />

            {simulatedDots(rows).map((dot) => (
              <circle key={dot.id} cx={x(dot.time)} cy={y(dot.value)} r="1.4" fill={GetColor(dot.map, dot.cir)} opacity="0.55" />
            ))}

            <path d={line(rows)} fill="none" stroke={selected ? "#242424" : "#757f84"} strokeWidth={selected ? 2 : 1.2} />

            {rows.filter((_, i) => i % 14 === 0).map((d, i) => (
              <circle key={`${category}-${+d.time}-${i}`} cx={x(d.time)} cy={y(Rating(d.map))} r={selected ? 2.3 : 1.7} fill={GetColor(d.map, d.cir)} stroke="#fff" strokeWidth="0.5" />
            ))}

            {Time && <line x1={x(Time)} x2={x(Time)} y1="8" y2={rowHeight - 15} stroke="#222" strokeWidth="1.3" />}

            <g transform={`translate(${width - margin.right},0)`}>
              {[0, 5, 10].map((tick) => (
                <g key={tick} transform={`translate(0,${y(tick)})`}>
                  <line x2="4" stroke="#999" />
                  <text x="7" y="3" fontSize="9" fill="#555">{tick}</text>
                </g>
              ))}
            </g>

            {selected && <rect x={margin.left - 4} y="4" width={width - margin.left - margin.right + 8} height={rowHeight - 18} fill="none" stroke="#444" opacity="0.25" />}

          </g>
        )
      })}

      <g transform={`translate(0, ${height - margin.bottom})`}>
        {x.ticks(7).map((tick) => (
          <g key={+tick} transform={`translate(${x(tick)},0)`}>
            <line y2="5" stroke="#999" />
            <text y="18" textAnchor="middle" fontSize="9" fill="#555">{d3.timeFormat("%a %d %H:%M")(tick)}</text>
          </g>
        ))}
        <line x1={margin.left} x2={width - margin.right} stroke="#ccc" />
      </g>
    </svg>
  )
}
