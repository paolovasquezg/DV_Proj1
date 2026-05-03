import * as d3 from "d3"
import { vsupColor } from "../utils/vsup"

export default function VSUPLegend() {
  const width = 280
  const height = 250
  const cx = 138
  const cy = 205
  const innerRadius = 35
  const ringSize = 32
  const angleStart = -70
  const angleEnd = 70

  const ratingSteps = d3.range(0, 10, 1)
  const cirTiers = [
    { min: 0, max: 1.25, label: "0 - 1.25" },
    { min: 1.25, max: 2.5, label: "1.25 - 2.50" },
    { min: 2.5, max: 5, label: "2.50 - 5.00" },
    { min: 5, max: 10, label: "> 5.00" }
  ]

  const angle = d3.scaleLinear().domain([0, 10]).range([angleStart, angleEnd])
  const arc = d3.arc()

  const cells = cirTiers.flatMap((tier, tierIndex) =>
    ratingSteps.map((rating) => ({
      rating,
      ratingMid: rating + 0.5,
      cirMid: (tier.min + tier.max) / 2,
      tierIndex,
      tier
    }))
  )

  return (
    <svg className="panel" viewBox={`0 0 ${width} ${height}`}>
      <rect width={width} height={height} fill="#fbfbfb" />

      <text x="18" y="20" fontSize="12" fontWeight="700">
        VSUP Fan
      </text>

      <text x="18" y="38" fontSize="10" fill="#666">
        Rating + 95% CIR
      </text>

      <g>
        {cells.map((d) => {
          const r0 = innerRadius + d.tierIndex * ringSize
          const r1 = r0 + ringSize - 1
          const a0 = (angle(d.rating) * Math.PI) / 180
          const a1 = (angle(d.rating + 1) * Math.PI) / 180

          return (
            <path
              key={`${d.tierIndex}-${d.rating}`}
              d={arc({
                innerRadius: r0,
                outerRadius: r1,
                startAngle: a0,
                endAngle: a1
              })}
              transform={`translate(${cx},${cy})`}
              fill={vsupColor(d.ratingMid, d.cirMid)}
              stroke="#ffffff"
              strokeWidth="0.4"
            >
              <title>{`Rating: ${d.rating} - ${d.rating + 1}
95% CIR: ${d.tier.label}`}</title>
            </path>
          )
        })}
      </g>

      {[0, 2.5, 5, 7.5, 10].map((value) => {
        const a = (angle(value) - 90) * Math.PI / 180
        const r = innerRadius + cirTiers.length * ringSize + 12
        const x = cx + Math.cos(a) * r
        const y = cy + Math.sin(a) * r

        return (
          <text key={value} x={x} y={y} fontSize="8" textAnchor="middle" fill="#555">
            {value}
          </text>
        )
      })}

      <text x={cx} y={height - 10} textAnchor="middle" fontSize="10" fill="#555">
        Rating
      </text>

      <g transform="translate(18,60)">
        <text x="0" y="0" fontSize="10" fontWeight="700" fill="#555">
          95% CIR
        </text>

        {cirTiers.map((tier, index) => (
          <g key={tier.label} transform={`translate(0,${18 + index * 18})`}>
            <rect x="0" y="-8" width="12" height="12" fill={vsupColor(7.5, (tier.min + tier.max) / 2)} />
            <text x="18" y="2" fontSize="9" fill="#555">
              {tier.label}
            </text>
          </g>
        ))}
      </g>
    </svg>
  )
}