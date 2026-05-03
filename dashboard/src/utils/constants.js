export const LABELS = {
  shake_intensity: "Shake Intensity", medical: "Medical", power: "Power",
  buildings: "Buildings", sewer_and_water: "Sewer & Water", roads_and_bridges: "Roads & Bridges"
}

export const NEIGHBOURHOODS = {
  1: "Palace Hills", 2: "Northwest", 3: "Old Town", 4: "Safe Town", 5: "Southwest", 6: "Downtown", 7: "Wilson Forest",
  8: "Scenic Vista", 9: "Broadview", 10: "Chapparal", 11: "Terrapin Springs", 12: "Pepper Mill", 13: "Cheddarford",
  14: "East Parton", 15: "Weston", 16: "Southton", 17: "Oak Willow", 18: "Easton", 19: "West Parton"
}

export const LOCATIONS = Array.from({ length: 19 }, (_, i) => i + 1)

export const ORDER = [
  "shake_intensity", "medical", "power",
  "buildings", "sewer_and_water", "roads_and_bridges"
]