import { useEffect } from "react"

export default function Timeline({
  times,
  selectedTimeIndex,
  setSelectedTimeIndex,
  isPlaying,
  setIsPlaying
}) {
  const selectedTime = times[selectedTimeIndex]

  useEffect(() => {
    if (!isPlaying || times.length === 0) return

    const id = setInterval(() => {
      setSelectedTimeIndex((current) => {
        if (current >= times.length - 1) return 0
        return current + 1
      })
    }, 450)

    return () => clearInterval(id)
  }, [isPlaying, times.length, setSelectedTimeIndex])

  return (
    <div className="timeline-card">
      <div className="timeline-main">
        <button className="play-button" onClick={() => setIsPlaying(!isPlaying)}>
          {isPlaying ? "Pause" : "Play"}
        </button>

        <input
          className="timeline-range"
          type="range"
          min="0"
          max={Math.max(0, times.length - 1)}
          value={selectedTimeIndex}
          onChange={(e) => setSelectedTimeIndex(Number(e.target.value))}
        />

        <div className="timeline-date">
          {selectedTime ? selectedTime.toLocaleString() : ""}
        </div>
      </div>
    </div>
  )
}