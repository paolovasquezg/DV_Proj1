import * as d3 from "d3"
import { vsupColor } from "../../utils/vsup"

const toRad = (deg) => (deg * Math.PI) / 180

export default function VSUPLegend() {
  const width = 300
  const height = 480
  const cx = 140
  const cy = 160

  const innerRadius = 0
  const ringSize = 28
  const numTiers = 4
  const outerRadius = innerRadius + numTiers * ringSize
  const arc = d3.arc()

  const cirTiers = [
    { cir: 10 },
    { cir: 7.5 },
    { cir: 3.75 },
    { cir: 1.25 },
  ]

  const angleDeg = d3.scaleLinear().domain([0, 10]).range([-60, 60])

  const pt = (r, deg) => ({
    x: cx + r * Math.sin(toRad(deg)),
    y: cy - r * Math.cos(toRad(deg)),
  })

  const ratingTicks = [0, 1.25, 2.5, 3.75, 5, 6.25, 7.5, 8.75, 10]

  const cirBoundaries = [
    { r: outerRadius, label: "0" },
    { r: innerRadius + ringSize * 3, label: "2.5" },
    { r: innerRadius + ringSize * 2, label: "5" },
    { r: innerRadius + ringSize, label: "7.5" },
    { r: innerRadius, label: "10" },
  ]

  const cirLabelAngle = 60

  // Legend items below the fan
  const legendY = cy + 60
  const swatchSize = 10
  const legendItems = [
    {
      color: vsupColor(8.5, 1.25),
      label: "High rating, low uncertainty",
    },
    {
      color: vsupColor(1.5, 1.25),
      label: "Low rating, low uncertainty",
    },
    {
      color: vsupColor(5, 10),
      label: "High uncertainty",
    },
  ]

  return (
    <svg className="w-full max-w-md block" viewBox={`0 0 ${width} ${height}`}>
      {/* Rating label */}
      <text x={cx} y={cy - outerRadius - 28} textAnchor="middle" fontSize="11" fontWeight="600" fill="#333">
        Rating
      </text>

      {/* Fan cells */}
      {cirTiers.flatMap((tier, ti) =>
        d3.range(0, 10).map((r) => {
          const r0 = innerRadius + ti * ringSize
          const r1 = r0 + ringSize - 1
          const a0 = toRad(angleDeg(r))
          const a1 = toRad(angleDeg(r + 1))
          return (
            <path
              key={`${ti}-${r}`}
              d={arc({ innerRadius: r0, outerRadius: r1, startAngle: a0, endAngle: a1 }) || ""}
              transform={`translate(${cx},${cy})`}
              fill={vsupColor(r + 0.5, tier.cir)}
              stroke="white"
              strokeWidth="0.5"
            />
          )
        })
      )}

      {/* Rating tick labels */}
      {ratingTicks.map((v) => {
        const { x, y } = pt(outerRadius + 12, angleDeg(v))
        return (
          <text key={v} x={x} y={y} textAnchor="middle" dominantBaseline="middle" fontSize="8" fill="#555">
            {v}
          </text>
        )
      })}

      {/* CIR boundary labels */}
      {cirBoundaries.map(({ r, label }) => {
        const { x, y } = pt(r, cirLabelAngle)
        return (
          <text key={label} x={x + 4} y={y + 10} textAnchor="start" dominantBaseline="middle" fontSize="8" fill="#555">
            {label}
          </text>
        )
      })}

      {/* CIR axis label */}
      {(() => {
        const midR = (innerRadius + outerRadius) / 2
        const { x, y } = pt(midR, cirLabelAngle)
        return (
          <text
            transform={`translate(${x + 22}, ${y + 20}) rotate(-30)`}
            textAnchor="middle" dominantBaseline="middle" fontSize="8" fill="#666"
          >
            95% Credible Interval Range (CIR)
          </text>
        )
      })()}

      {/* Divider */}
      <line
        x1={16} y1={legendY - 14}
        x2={width - 16} y2={legendY - 14}
        stroke="#ddd" strokeWidth="0.5"
      />

      {/* Legend title */}
      <text x={16} y={legendY} fontSize="9" fontWeight="600" fill="#444">
        How to read this chart
      </text>

      {/* Legend items */}
      {legendItems.map(({ color, label }, i) => (
        <g key={i} transform={`translate(16, ${legendY + 18 + i * 22})`}>
          <rect width={swatchSize} height={swatchSize} rx="2" fill={color} />
          <text x={swatchSize + 6} y={swatchSize / 2} dominantBaseline="middle" fontSize="8.5" fill="#555">
            {label}
          </text>
        </g>
      ))}

    </svg>
  )
}