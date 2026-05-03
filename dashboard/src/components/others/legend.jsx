import * as d3 from "d3"
import { GetColor } from "../../utils/vsup"

export default function Legend() {

  const toRad = (deg) => (deg * Math.PI) / 180
  const pt = (r, deg) => ({ x: cx + r * Math.sin(toRad(deg)), y: cy - r * Math.cos(toRad(deg)) })

  const width = 300
  const height = 480
  const cx = 140
  const cy = 172

  const innerRadius = 0
  const ringSize = 28
  const numTiers = 4
  const outerRadius = innerRadius + numTiers * ringSize
  const arc = d3.arc()

  const angleDeg = d3.scaleLinear().domain([0, 10]).range([-60, 60])

  const Tiers = [{ cir: 10 }, { cir: 7.5 }, { cir: 3.75 }, { cir: 1.25 }]

  const RatingTicks = [0, 2.5, 5, 7.5, 10]

  const CirBoundaries = [{ r: outerRadius, label: "0" }, { r: innerRadius + ringSize * 3, label: "2.5" },
  { r: innerRadius + ringSize * 2, label: "5" }, { r: innerRadius + ringSize, label: "7.5" },
  { r: innerRadius, label: "10" }]

  const cirLabelAngle = 60
  const midR = (innerRadius + outerRadius) / 2

  const legendY = cy + 60
  const swatchSize = 10

  const LegendItems = [
    { color: GetColor(8.5, 1.25), label: "High rating, low uncertainty" },
    { color: GetColor(5, 10), label: "High uncertainty" },
    { color: GetColor(1.5, 1.25), label: "Low rating, low uncertainty" }]

  return (
    <svg className="w-full max-w-md block" viewBox={`0 0 ${width} ${height}`}>

      <text x={cx} y={cy - outerRadius - 28} textAnchor="middle" fontSize="11" fontWeight="600" fill="#333">
        Rating
      </text>

      {Tiers.flatMap((tier, ti) =>
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
              fill={GetColor(r + 0.5, tier.cir)}
              stroke="white"
              strokeWidth="0.5"
            />
          )
        })
      )}

      {RatingTicks.map((v) => {
        const { x, y } = pt(outerRadius + 12, angleDeg(v))
        return (
          <g key={v}>
            <text x={x} y={y} textAnchor="middle" dominantBaseline="middle" fontSize="8" fill="#555">{v}</text>
          </g>
        )
      })}

      {CirBoundaries.map(({ r, label, note }) => {
        const { x, y } = pt(r, cirLabelAngle)
        return (
          <g key={label}>
            <text x={x + 4} y={y + 10} textAnchor="start" dominantBaseline="middle" fontSize="8" fill="#555">{label}</text>
          </g>
        )
      })}

      {(() => {
        const { x, y } = pt(midR, cirLabelAngle)
        return (
          <text transform={`translate(${x + 22}, ${y + 20}) rotate(-30)`} textAnchor="middle" dominantBaseline="middle" fontSize="8" fill="#666">
            95% Credible Interval Range (CIR)
          </text>
        )
      })()}

      <line x1={16} y1={legendY - 14} x2={width - 16} y2={legendY - 14} stroke="#ddd" strokeWidth="0.5" />

      <text x={16} y={legendY} fontSize="9" fontWeight="600" fill="#444">
        Examples
      </text>

      {LegendItems.map(({ color, label }, i) => (
        <g key={i} transform={`translate(16, ${legendY + 18 + i * 22})`}>
          <rect width={swatchSize} height={swatchSize} rx="2" fill={color} />
          <text x={swatchSize + 6} y={swatchSize / 2} dominantBaseline="middle" fontSize="8.5" fill="#555">{label}</text>
        </g>
      ))}

    </svg>
  )
}
