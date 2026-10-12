// ============================================================
// HRRT Dhaka | STEP 3a: Export DEM (for natural drainage network)
// The DEM is exported at 30 m in UTM 46N so it matches the Sentinel-2 export.
// ============================================================

var aoi = ee.Geometry.Rectangle([90.28, 23.62, 90.52, 23.92]);
Map.centerObject(aoi, 11);

// Copernicus GLO-30 DEM (free). Restore projection lost by mosaic().
var demCol = ee.ImageCollection('COPERNICUS/DEM/GLO30')
  .filterBounds(aoi)
  .select('DEM');
var proj = demCol.first().select(0).projection();
var dem = demCol.mosaic().setDefaultProjection(proj).clip(aoi);

print('DEM min/max (m):', dem.reduceRegion({
  reducer: ee.Reducer.minMax(), geometry: aoi, scale: 30,
  maxPixels: 1e9, bestEffort: true
}));
Map.addLayer(dem, {min: 0, max: 20,
  palette: ['000080', '00ffff', '00ff00', 'ffff00', 'ff0000']}, 'DEM (m)');

Export.image.toDrive({
  image: dem.toFloat(),
  description: 'Dhaka_DEM_GLO30_30m',
  folder: 'HRRT_Dhaka',
  region: aoi,
  scale: 30,
  crs: 'EPSG:32646',
  maxPixels: 1e9
});
