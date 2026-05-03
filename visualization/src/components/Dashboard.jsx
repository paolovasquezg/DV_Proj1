import { useEffect, useMemo, useState } from "react"
import { loadMc1Excel } from "../utils/loadExcel"
import { CATEGORY_LABELS, LOCATIONS, NEIGHBOURHOOD_NAMES } from "../utils/constants"
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

  useEffect(() => {
    loadMc1Excel()
      .then((rows) => {
        setData(rows)
        setLoading(false)
      })
      .catch((error) => {
        console.error(error)
        setLoading(false)
      })
  }, [])

  const times = useMemo(() => getTimes(data), [data])
  const categories = useMemo(() => getCategories(data), [data])

  useEffect(() => {
    if (times.length > 0 && selectedTimeIndex === 0) {
      setSelectedTimeIndex(Math.floor(times.length * 0.65))
    }
  }, [times, selectedTimeIndex])

  useEffect(() => {
    if (categories.length > 0 && !categories.includes(selectedCategory)) {
      setSelectedCategory(categories[0])
    }
  }, [categories, selectedCategory])

  const selectedTime = times[selectedTimeIndex]

  const currentRows = useMemo(() => {
    if (!selectedTime) return []
    return latestRowsFor(data, selectedCategory, selectedTime)
  }, [data, selectedCategory, selectedTime])

  const selectedRow = currentRows.find((d) => d.location === selectedLocation)

  if (loading) {
    return <div className="loading">Loading MC1 dashboard...</div>
  }

  if (!data.length) {
    return <div className="loading">No data loaded.</div>
  }

  return (
    <div className="dashboard-page">
      <header className="dashboard-header">
        <div>
          <h1>VAST Challenge 2019 MC1 Simulation</h1>
          <p>BSTS MAP, credible intervals and CIR visualization using React + D3.</p>
        </div>
      </header>

      <Controls
        categories={categories}
        locations={LOCATIONS}
        selectedCategory={selectedCategory}
        setSelectedCategory={setSelectedCategory}
        selectedLocation={selectedLocation}
        setSelectedLocation={setSelectedLocation}
        sortMode={sortMode}
        setSortMode={setSortMode}
        fillMap={fillMap}
        setFillMap={setFillMap}
        showNames={showNames}
        setShowNames={setShowNames}
      />

      <Timeline
        times={times}
        selectedTimeIndex={selectedTimeIndex}
        setSelectedTimeIndex={setSelectedTimeIndex}
        isPlaying={isPlaying}
        setIsPlaying={setIsPlaying}
      />

      <main className="dashboard-grid">
        <section className="panel-card">
          <div className="panel-title">
            <div>
              <h2>Map with Choropleth and Circle Marks</h2>
              <span>{CATEGORY_LABELS[selectedCategory]}</span>
            </div>
          </div>

          <ChoroplethMap
            rows={currentRows}
            selectedLocation={selectedLocation}
            setSelectedLocation={setSelectedLocation}
            fillMap={fillMap}
            showNames={showNames}
          />
        </section>

        <section className="panel-card">
          <div className="panel-title">
            <div>
              <h2>Error Bar Chart</h2>
              <span>MAP + 50/80/95% credible intervals</span>
            </div>
          </div>

          <ErrorBarChart
            rows={currentRows}
            selectedLocation={selectedLocation}
            sortMode={sortMode}
          />
        </section>
      </main>

      <section className="middle-grid">
        <div className="info-card">
          <span>Neighbourhood:</span>
          <strong>{selectedLocation} {NEIGHBOURHOOD_NAMES[selectedLocation]}</strong>

          <div className="info-metrics">
            <div>
              <span>Category</span>
              <b>{CATEGORY_LABELS[selectedCategory]}</b>
            </div>

            <div>
              <span>Time</span>
              <b>{selectedTime ? selectedTime.toLocaleString() : ""}</b>
            </div>

            <div>
              <span>MAP</span>
              <b>{Number.isFinite(selectedRow?.map) ? selectedRow.map.toFixed(2) : "No data"}</b>
            </div>

            <div>
              <span>95% CIR</span>
              <b>{Number.isFinite(selectedRow?.cir) ? selectedRow.cir.toFixed(2) : "No data"}</b>
            </div>
          </div>

          <p>
            The heat map displays hourly aggregated MAP values and their uncertainty.
            The line charts show the progression of MAP values and credible intervals over time.
          </p>
        </div>

        <section className="panel-card">
          <HeatMap
            data={data}
            selectedLocation={selectedLocation}
            selectedTime={selectedTime}
            selectedCategory={selectedCategory}
            setSelectedCategory={setSelectedCategory}
          />
        </section>

        <section className="panel-card">
          <VSUPLegend />
        </section>
      </section>

      <section className="panel-card line-section">
        <LineCharts
          data={data}
          selectedLocation={selectedLocation}
          selectedTime={selectedTime}
          selectedCategory={selectedCategory}
          setSelectedCategory={setSelectedCategory}
        />
      </section>
    </div>
  )
}