import { useMemo } from "react"
import * as d3 from "d3"
import { CATEGORY_LABELS, CATEGORY_ORDER } from "../utils/constants"
import { vsupColor } from "../utils/vsup"

export default function HeatMap({
  data,
  selectedLocation,
  selectedTime,
  selectedCategory,
  setSelectedCategory
}) {
  const width = 900
  const height = 185
  const margin = { top: 24, right: 18, bottom: 32, left: 125 }

  const locationData = useMemo(() => {
    return data.filter((d) => d.location === selectedLocation)
  }, [data, selectedLocation])

  const categories = useMemo(() => {
    const existing = new Set(locationData.map((d) => d.category))
    return CATEGORY_ORDER.filter((category) => existing.has(category))
  }, [locationData])

  const hourlyData = useMemo(() => {
    const grouped = d3.rollups(
      locationData,
      (values) => {
        const maxRow = values.reduce((best, current) => {
          if (!best) return current
          return current.map > best.map ? current : best
        }, null)

        return {
          ...maxRow,
          hour: d3.timeHour.floor(maxRow.time)
        }
      },
      (d) => +d3.timeHour.floor(d.time),
      (d) => d.category
    )

    return grouped.flatMap(([hourKey, categoryGroups]) =>
      categoryGroups.map(([, row]) => ({
        ...row,
        hour: new Date(Number(hourKey))
      }))
    )
  }, [locationData])

  const timeExtent = d3.extent(locationData, (d) => d.time)
  const x = d3.scaleTime().domain(timeExtent).range([margin.left, width - margin.right])
  const y = d3.scaleBand().domain(categories).range([margin.top, height - margin.bottom]).padding(0.08)

  const hours = Array.from(new Set(hourlyData.map((d) => +d.hour))).sort(d3.ascending)
  const cellWidth = hours.length > 1 ? Math.max(2, x(new Date(hours[1])) - x(new Date(hours[0]))) : 6

  return (
    <svg className="block rounded-xl overflow-hidden" viewBox={`0 0 ${width} ${height}`}>
      <rect width={width} height={height} fill="white" />

      {categories.map((category) => {
        const selected = category === selectedCategory

        return (
          <g key={category} onClick={() => setSelectedCategory(category)} style={{ cursor: "pointer" }}>
            <text
              x={margin.left - 10}
              y={y(category) + y.bandwidth() / 2 + 4}
              textAnchor="end"
              fontSize="10"
              fontWeight={selected ? "700" : "400"}
              fill={selected ? "#222" : "#555"}
            >
              {CATEGORY_LABELS[category] ?? category}
            </text>

            {selected && (
              <rect
                x={margin.left - 4}
                y={y(category) - 2}
                width={width - margin.left - margin.right + 8}
                height={y.bandwidth() + 4}
                fill="none"
                stroke="#555"
                strokeWidth="1"
                opacity="0.35"
              />
            )}
          </g>
        )
      })}

      {hourlyData.map((d, index) => (
        <rect
          key={`${+d.hour}-${d.category}-${index}`}
          x={x(d.hour)}
          y={y(d.category)}
          width={cellWidth}
          height={y.bandwidth()}
          fill={vsupColor(d.map, d.cir)}
        >
          <title>
            {`${CATEGORY_LABELS[d.category] ?? d.category}
${d.hour.toLocaleString()}
MAP: ${d.map?.toFixed?.(2)}
CIR: ${d.cir?.toFixed?.(2)}`}
          </title>
        </rect>
      ))}

      {selectedTime && (
        <line
          x1={x(selectedTime)}
          x2={x(selectedTime)}
          y1={margin.top - 6}
          y2={height - margin.bottom}
          stroke="#222"
          strokeWidth="1.5"
        />
      )}

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