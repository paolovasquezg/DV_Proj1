import { CATEGORY_LABELS, NEIGHBOURHOOD_NAMES } from "../utils/constants"

export default function Controls({
  categories,
  locations,
  selectedCategory,
  setSelectedCategory,
  selectedLocation,
  setSelectedLocation,
  sortMode,
  setSortMode,
  fillMap,
  setFillMap,
  showNames,
  setShowNames
}) {
  return (
    <div className="controls">
      <div className="control-row">
        <div className="control-group">
          <label>Category</label>
          <select value={selectedCategory} onChange={(e) => setSelectedCategory(e.target.value)}>
            {categories.map((category) => (
              <option key={category} value={category}>
                {CATEGORY_LABELS[category] ?? category}
              </option>
            ))}
          </select>
        </div>

        <div className="control-group">
          <label>Neighbourhood</label>
          <select value={selectedLocation} onChange={(e) => setSelectedLocation(Number(e.target.value))}>
            {locations.map((location) => (
              <option key={location} value={location}>
                {location} {NEIGHBOURHOOD_NAMES[location]}
              </option>
            ))}
          </select>
        </div>

        <div className="control-group">
          <label>Error Bar Chart sort options</label>
          <select value={sortMode} onChange={(e) => setSortMode(e.target.value)}>
            <option value="ci95">95% CI lower bound</option>
            <option value="map">Rating MAP</option>
            <option value="cir">95% CIR</option>
            <option value="location">Neighbourhood</option>
          </select>
        </div>
      </div>

      <div className="toggle-row">
        <label>
          <input type="checkbox" checked={fillMap} onChange={(e) => setFillMap(e.target.checked)} />
          Fill neighbourhoods with colours
        </label>

        <label>
          <input type="checkbox" checked={showNames} onChange={(e) => setShowNames(e.target.checked)} />
          Show neighbourhood names
        </label>
      </div>
    </div>
  )
}