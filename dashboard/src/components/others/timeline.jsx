import { useEffect } from "react"

export default function Timeline({ times, TimeIndex, setTimeIndex, Play }) {
  const selectedTime = times[TimeIndex]

  useEffect(() => {
    if (!Play) return
    const id = setInterval(() => { setTimeIndex((current) => { if (current >= times.length - 1) return 0; return current + 1 }) }, 450)
    return () => clearInterval(id)

  }, [Play, times.length, setTimeIndex])

  return (
    <div className="bg-white border border-gray-200 rounded-2xl px-4 py-2.5 mb-2.5 shadow-sm">
      <div className="flex items-center gap-3.5">
        <input
          type="range"
          min="0"
          max={Math.max(0, times.length - 1)}
          value={TimeIndex}
          onChange={(e) => setTimeIndex(Number(e.target.value))}
          className="flex-1 cursor-pointer" />

        <div className="text-xs font-semibold text-gray-700 whitespace-nowrap min-w-[190px] text-right">
          {selectedTime?.toLocaleString()}
        </div>

      </div>
    </div>
  )
}
