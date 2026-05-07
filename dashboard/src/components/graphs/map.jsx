import { CountyShapes, CountyForm } from "../../utils/shapes"
import { NEIGHBOURHOODS } from "../../utils/constants"
import { StatusColor, GetColor } from "../../utils/vsup"

export default function MainMap({ Regs, Location, setLocation, fillMap, showNames }) {
  const width = 610
  const height = 560
  const ByLocation = new Map(Regs.map((d) => [d.location, d]))

  return (
    <svg className="block w-full rounded-xl overflow-hidden" viewBox={`0 0 ${width} ${height}`}>
      <rect width={width} height={height} fill="white" />

      <g transform="translate(25,55)">
        {CountyShapes.map((shape) => {
          const reg = ByLocation.get(shape.id)
          const selected = shape.id === Location

          return (
            <g key={shape.id} onClick={() => setLocation(shape.id)} style={{ cursor: "pointer" }}>
              <path
                d={CountyForm(shape.p)}
                fill={selected ? "#f4a77d" : fillMap ? GetColor(reg?.map, reg?.cir) : "#dce5e8"}
                stroke={selected ? "#555" : "#aab5ba"}
                strokeWidth={selected ? 2 : 0.8}
                strokeDasharray={reg?.status === "missing" ? "2 2" : ""}
              />
              <circle cx={shape.c[0]} cy={shape.c[1]} r={selected ? 11 : 9} fill="none" stroke={StatusColor[reg?.status ?? "missing"]} strokeWidth="4" />
              <circle cx={shape.c[0]} cy={shape.c[1]} r="4" fill="#ffffff" />
              <text x={shape.c[0]} y={shape.c[1] - 16} textAnchor="middle" fontSize="10" fontWeight="700" fill="#333">
                {shape.id}
              </text>
              {showNames && (
                <text x={shape.c[0]} y={shape.c[1] + 25} textAnchor="middle" fontSize="9" fill="#555">
                  {NEIGHBOURHOODS[shape.id]}
                </text>
              )}
            </g>
          )
        })}
      </g>

      <g transform="translate(42,450)">
        <circle cx="0" cy="0" r="8" fill="none" stroke={StatusColor.fresh} strokeWidth="4" />
        <text x="18" y="4" fontSize="10" fill="#666">Fresh data within 15 minutes</text>

        <circle cx="0" cy="24" r="8" fill="none" stroke={StatusColor.old} strokeWidth="4" />
        <text x="18" y="28" fontSize="10" fill="#666">Old data within 1 hour</text>

        <circle cx="0" cy="48" r="8" fill="none" stroke={StatusColor.missing} strokeWidth="4" />
        <text x="18" y="52" fontSize="10" fill="#666">Missing data over 1 hour</text>
      </g>

      <g transform="translate(475,480)">
        <line x1="0" x2="70" y1="0" y2="0" stroke="#aaa" strokeWidth="2" />
        <line x1="0" x2="0" y1="-5" y2="5" stroke="#aaa" />
        <line x1="70" x2="70" y1="-5" y2="5" stroke="#aaa" />
        <text x="48" y="-7" fontSize="10" fill="#777">3 mi</text>
      </g>
    </svg>
  )
}
