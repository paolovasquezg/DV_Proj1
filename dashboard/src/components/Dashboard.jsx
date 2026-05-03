import { useEffect, useMemo, useState } from "react"
import { loadMc1Excel } from "../utils/loadExcel"
import { CATEGORY_LABELS, CATEGORY_ORDER, LOCATIONS, NEIGHBOURHOOD_NAMES } from "../utils/constants"
import { getCategories, getTimes, latestRowsFor } from "../utils/selectors"
import Controls from "./Controls"
import Timeline from "./Timeline"
import ErrorBarChart from "./ErrorBarChart"
import ChoroplethMap from "./ChoroplethMap"
import HeatMap from "./HeatMap"
import LineCharts from "./LineCharts"
import VSUPLegend from "./VSUPLegend"

export default function Dashboard() {
  const [data, setData] = useState([])
  const [loading, setLoading] = useState(true)
  const [selectedTimeIndex, setSelectedTimeIndex] = useState(0)
  const [selectedCategory, setSelectedCategory] = useState("shake_intensity")
  const [selectedLocation, setSelectedLocation] = useState(4)
  const [sortMode, setSortMode] = useState("ci95")
  const [fillMap, setFillMap] = useState(true)
  const [showNames, setShowNames] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [palette, setPalette] = useState("vsup")

  useEffect(() => {
    loadMc1Excel()
      .then((rows) => { setData(rows); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  const times = useMemo(() => getTimes(data), [data])
  const categories = useMemo(() => getCategories(data), [data])

  useEffect(() => {
    if (times.length > 0 && selectedTimeIndex === 0)
      setSelectedTimeIndex(Math.floor(times.length * 0.65))
  }, [times, selectedTimeIndex])

  useEffect(() => {
    if (categories.length > 0 && !categories.includes(selectedCategory))
      setSelectedCategory(categories[0])
  }, [categories, selectedCategory])

  const selectedTime = times[selectedTimeIndex]

  const currentRows = useMemo(() => {
    if (!selectedTime) return []
    return latestRowsFor(data, selectedCategory, selectedTime)
  }, [data, selectedCategory, selectedTime])

  if (loading) return <div className="p-10 text-center text-sm text-gray-500">Loading MC1 dashboard...</div>
  if (!data.length) return <div className="p-10 text-center text-sm text-gray-500">No data loaded.</div>

  const orderedCategories = CATEGORY_ORDER.filter((c) => categories.includes(c))

  return (
    <div className="max-w-[1380px] mx-auto px-5 py-4">

      {/* ── Top: map (left) + category buttons + error bar chart (right) ── */}
      <div className="grid gap-3 mb-2.5 items-start" style={{ gridTemplateColumns: "minmax(480px, 540px) 1fr" }}>
        <section className="flex items-center justify-center">
          <ChoroplethMap
            rows={currentRows}
            selectedLocation={selectedLocation}
            setSelectedLocation={setSelectedLocation}
            fillMap={fillMap}
            showNames={showNames}
          />
        </section>

        <div className="flex flex-col gap-2">
          <div className="grid grid-cols-3 gap-1.5">
            {orderedCategories.map((cat) => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={`h-8 border rounded-lg text-xs font-medium cursor-pointer transition-colors whitespace-nowrap ${selectedCategory === cat
                  ? "bg-sky-400 border-sky-400 text-white"
                  : "bg-white border-gray-300 text-slate-600 hover:bg-slate-50"
                  }`}
              >
                {CATEGORY_LABELS[cat]}
              </button>
            ))}
          </div>
          <ErrorBarChart
            rows={currentRows}
            selectedLocation={selectedLocation}
            setSelectedLocation={setSelectedLocation}
            sortMode={sortMode}
          />
        </div>
      </div>

      {/* ── Timeline ── */}
      <Timeline
        times={times}
        selectedTimeIndex={selectedTimeIndex}
        setSelectedTimeIndex={setSelectedTimeIndex}
        isPlaying={isPlaying}
        setIsPlaying={setIsPlaying}
      />

      {/* ── Controls ── */}
      <Controls
        locations={LOCATIONS}
        selectedLocation={selectedLocation}
        setSelectedLocation={setSelectedLocation}
        sortMode={sortMode}
        setSortMode={setSortMode}
        fillMap={fillMap}
        setFillMap={setFillMap}
        showNames={showNames}
        setShowNames={setShowNames}
        isPlaying={isPlaying}
        setIsPlaying={setIsPlaying}
        palette={palette}
        setPalette={setPalette}
      />

      {/* ── Bottom: heatmap + linecharts (left) | vsup legend (right) ── */}
      <div className="grid gap-3 items-start" style={{ gridTemplateColumns: "3fr 2fr" }}>

        <div className="flex flex-col gap-3">
          <HeatMap
            data={data}
            selectedLocation={selectedLocation}
            selectedTime={selectedTime}
            selectedCategory={selectedCategory}
            setSelectedCategory={setSelectedCategory}
          />

          <div className="overflow-y-auto max-h-[300px]">
            <LineCharts
              data={data}
              selectedLocation={selectedLocation}
              selectedTime={selectedTime}
              selectedCategory={selectedCategory}
              setSelectedCategory={setSelectedCategory}
            />
          </div>
        </div>

        <VSUPLegend />

      </div>

    </div>
  )
}
