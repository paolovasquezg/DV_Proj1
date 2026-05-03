import { NEIGHBOURHOOD_NAMES } from "../utils/constants"

export default function Controls({
  locations,
  selectedLocation,
  setSelectedLocation,
  sortMode,
  setSortMode,
  fillMap,
  setFillMap,
  showNames,
  setShowNames,
  isPlaying,
  setIsPlaying,
  palette,
  setPalette
}) {
  return (
    <div className="px-4 py-2.5 mb-3 grid items-center gap-30"
      style={{ gridTemplateColumns: "auto 1fr auto" }}>

      {/* Left: toggles */}
      <div className="flex flex-row gap-5">
        <label className="flex items-center gap-1.5 text-xs text-slate-600 cursor-pointer whitespace-nowrap">
          <input type="checkbox" checked={fillMap} onChange={(e) => setFillMap(e.target.checked)} />
          Fill neighbourhoods with colours
        </label>
        <label className="flex items-center gap-1.5 text-xs text-slate-600 cursor-pointer whitespace-nowrap">
          <input type="checkbox" checked={showNames} onChange={(e) => setShowNames(e.target.checked)} />
          Show neighbourhood names
        </label>
      </div>

      {/* Center: play + neighbourhood */}
      <div className="flex items-center gap-2 justify-center">
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className={`h-8 px-3.5 rounded-lg text-sm font-bold cursor-pointer whitespace-nowrap border ${isPlaying
            ? "bg-white border-gray-300 text-gray-700"
            : "bg-sky-400 border-sky-400 text-white"
            }`}
        >
          {isPlaying ? "|| Pause" : "Play ▶"}
        </button>
        <select
          value={selectedLocation}
          onChange={(e) => setSelectedLocation(Number(e.target.value))}
          className="h-8 border border-gray-300 rounded-lg px-2.5 bg-white text-gray-700 text-xs"
        >
          {locations.map((loc) => (
            <option key={loc} value={loc}>
              {loc} {NEIGHBOURHOOD_NAMES[loc]}
            </option>
          ))}
        </select>
      </div>

      {/* Right: sort options + palette */}
      <div className="flex items-center gap-2 flex-nowrap">
        <span className="text-[11px] text-slate-400 whitespace-nowrap">Sort options</span>
        <select
          value={sortMode}
          onChange={(e) => setSortMode(e.target.value)}
          className="h-8 border border-gray-300 rounded-lg px-2 bg-white text-gray-700 text-xs"
        >
          <option value="ci95">95% CI lower bound</option>
          <option value="map">Rating MAP</option>
          <option value="cir">95% CIR</option>
          <option value="location">Neighbourhood</option>
        </select>
      </div>
    </div>
  )
}
