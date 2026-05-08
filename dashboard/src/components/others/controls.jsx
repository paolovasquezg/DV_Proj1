import { NEIGHBOURHOODS } from "../../utils/constants"

export default function Controls({
  locations,
  Location,
  setLocation,
  sort,
  setSort,
  fillMap,
  setFillMap,
  showAllCategories,
  setShowAllCategories,
  showHighlight,
  setShowHighlight,
  showNames,
  setShowNames,
  showHospitals,
  setShowHospitals,
  palette,
  setPalette,
  Play,
  setPlay,
}) {
  return (
    <div className="px-4 py-2.5 mb-3 grid items-center gap-8"
      style={{ gridTemplateColumns: "auto 1fr auto" }}>

      <div className="flex flex-col gap-1">
        <label className="flex items-center gap-1.5 text-xs cursor-pointer whitespace-nowrap text-sky-600 font-medium">
          <input type="checkbox" checked={fillMap} onChange={(e) => setFillMap(e.target.checked)} className="accent-sky-400" />
          Fill neighbourhoods with colours
        </label>
        <label className="flex items-center gap-1.5 text-xs text-slate-600 cursor-pointer whitespace-nowrap">
          <input type="checkbox" checked={showAllCategories} onChange={(e) => setShowAllCategories(e.target.checked)} />
          Show all categories on the map
        </label>
        <label className="flex items-center gap-1.5 text-xs text-slate-600 cursor-pointer whitespace-nowrap">
          <input type="checkbox" checked={showHighlight} onChange={(e) => setShowHighlight(e.target.checked)} />
          Highlight the selected neighbourhood
        </label>
        <label className="flex items-center gap-1.5 text-xs text-slate-600 cursor-pointer whitespace-nowrap">
          <input type="checkbox" checked={showNames} onChange={(e) => setShowNames(e.target.checked)} />
          Show neighbourhood names
        </label>
        <label className="flex items-center gap-1.5 text-xs text-slate-600 cursor-pointer whitespace-nowrap">
          <input type="checkbox" checked={showHospitals} onChange={(e) => setShowHospitals(e.target.checked)} />
          Show hospitals and the nuclear plant
        </label>
      </div>

      <div className="flex items-center gap-2 justify-center">
        <button
          onClick={() => setPlay(!Play)}
          className={`h-8 px-3.5 rounded-lg text-sm font-bold cursor-pointer whitespace-nowrap border ${Play
            ? "bg-white border-gray-300 text-gray-700"
            : "bg-sky-400 border-sky-400 text-white"
            }`}
        >
          {Play ? "|| Pause" : "Play ▶"}
        </button>
        <select
          value={Location}
          onChange={(e) => setLocation(Number(e.target.value))}
          className="h-8 border border-gray-300 rounded-lg px-2.5 bg-white text-gray-700 text-xs"
        >
          {locations.map((loc) => (
            <option key={loc} value={loc}>
              {loc} {NEIGHBOURHOODS[loc]}
            </option>
          ))}
        </select>
      </div>

      <div className="flex items-center gap-2 flex-wrap justify-end">
        <span className="text-[11px] text-slate-400 whitespace-nowrap">Error Bar Chart sort options</span>
        <select
          value={sort}
          onChange={(e) => setSort(e.target.value)}
          className="h-8 border border-gray-300 rounded-lg px-2 bg-white text-gray-700 text-xs"
        >
          <option value="ci95">95% CI lower bound</option>
          <option value="map">MAP</option>
          <option value="cir">CIR</option>
          <option value="location">Neighbourhood</option>
        </select>
        <span className="text-[11px] text-slate-400 whitespace-nowrap ml-1">Colour palette</span>
        {[
          { key: "vsup", label: "VSUP" },
          { key: "vsup_ext", label: "VSUP Extended" },
          { key: "normal", label: "Normal" },
        ].map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setPalette(key)}
            className={`h-8 px-3 rounded-lg text-xs font-medium cursor-pointer border whitespace-nowrap
              ${palette === key
                ? "bg-sky-500 border-sky-500 text-white"
                : "bg-white border-gray-300 text-slate-600 hover:bg-slate-50"}`}
          >
            {label}
          </button>
        ))}
      </div>
    </div>
  )
}
