import { useEffect, useMemo, useState } from "react"
import { load_excel } from "../utils/excel"

import { LABELS, ORDER, LOCATIONS } from "../utils/constants"
import { GetTimes, GetLatestRegs, GetAllCategoryRegs } from "../utils/selectors"

import BarChart from "./graphs/barchart"
import MainMap from "./graphs/map"
import HeatMap from "./graphs/heatmap"
import Lines from "./graphs/lines"

import Timeline from "./others/Timeline"
import Controls from "./others/controls"
import Legend from "./others/legend"

export default function Dashboard() {
  const [data, setData] = useState([])
  const [loading, setLoading] = useState(true)

  const [TimeIndex, setTimeIndex] = useState(0)
  const [Location, setLocation] = useState(1)

  const [sort, setSort] = useState("ci95")
  const [Category, setCategory] = useState("shake_intensity")

  const [fillMap, setFillMap] = useState(true)
  const [showAllCategories, setShowAllCategories] = useState(false)
  const [showHighlight, setShowHighlight] = useState(true)
  const [showNames, setShowNames] = useState(false)
  const [showHospitals, setShowHospitals] = useState(false)
  const [palette, setPalette] = useState("vsup")
  const [Play, setPlay] = useState(false)

  const times = useMemo(() => GetTimes(data), [data])
  const Time = times[TimeIndex]

  const Regs = useMemo(() => GetLatestRegs(data, Category, Time), [data, Category, Time])
  const AllRegs = useMemo(() => showAllCategories ? GetAllCategoryRegs(data, Time) : null, [data, Time, showAllCategories])

  const MapRegs = showAllCategories && AllRegs ? AllRegs : Regs

  useEffect(() => { load_excel().then((rows) => { setData(rows); setLoading(false) }).catch(() => setLoading(false)) }, [])
  useEffect(() => { if (times.length > 0 && TimeIndex === 0) setTimeIndex(Math.floor(times.length * 0.65)) }, [times, TimeIndex])

  if (loading) return <div className="p-10 text-center text-sm text-gray-500">Loading dashboard...</div>
  if (!data.length) return <div className="p-10 text-center text-sm text-gray-500">No data loaded.</div>

  return (
    <div className="max-w-[1380px] mx-auto px-5 py-4">
      <div className="grid gap-3 mb-2.5 items-start" style={{ gridTemplateColumns: "minmax(480px, 540px) 1fr" }}>
        <section className="flex items-center justify-center">
          <MainMap
            Regs={MapRegs}
            Location={Location}
            setLocation={setLocation}
            fillMap={fillMap}
            showNames={showNames}
            showHighlight={showHighlight}
            showHospitals={showHospitals}
            palette={palette} />
        </section>

        <div className="flex flex-col gap-2">
          <div className="grid grid-cols-3 gap-1.5">
            {ORDER.map((cat) => (
              <button
                key={cat}
                onClick={() => setCategory(cat)}
                className={`h-8 border rounded-lg text-xs font-medium cursor-pointer transition-colors whitespace-nowrap
                  ${Category === cat ? "bg-sky-400 border-sky-400 text-white" : "bg-white border-gray-300 text-slate-600 hover:bg-slate-50"}`}>
                {LABELS[cat]}
              </button>
            ))}
          </div>
          <BarChart
            regs={Regs}
            Location={Location}
            setLocation={setLocation}
            sort={sort}
            showHighlight={showHighlight}
            palette={palette} />
        </div>
      </div>

      <Timeline
        times={times}
        TimeIndex={TimeIndex}
        setTimeIndex={setTimeIndex}
        Play={Play} />

      <Controls
        locations={LOCATIONS}
        Location={Location}
        setLocation={setLocation}
        sort={sort}
        setSort={setSort}
        fillMap={fillMap}
        setFillMap={setFillMap}
        showAllCategories={showAllCategories}
        setShowAllCategories={setShowAllCategories}
        showHighlight={showHighlight}
        setShowHighlight={setShowHighlight}
        showNames={showNames}
        setShowNames={setShowNames}
        showHospitals={showHospitals}
        setShowHospitals={setShowHospitals}
        palette={palette}
        setPalette={setPalette}
        Play={Play}
        setPlay={setPlay} />

      <div className="grid gap-3 items-start" style={{ gridTemplateColumns: "3fr 2fr" }}>
        <div className="flex flex-col gap-3">
          <HeatMap
            data={data}
            Location={Location}
            Time={Time}
            Category={Category}
            setCategory={setCategory} />

          <div className="overflow-y-auto max-h-[300px]">
            <Lines
              data={data}
              Location={Location}
              Time={Time}
              Category={Category}
              setCategory={setCategory} />
          </div>
        </div>

        <Legend palette={palette} />
      </div>
    </div>
  )
}
