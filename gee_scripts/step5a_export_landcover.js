// ============================================================
// HRRT Dhaka | STEP 5a: Export current land cover (Dynamic World, 10 m)
// Mean class probabilities for the 2026 dry season (Jan-Mar), used in Step 5b
// to judge whether the land along each priority corridor is open enough to restore.
// ============================================================

var aoi = ee.Geometry.Rectangle([90.28, 23.62, 90.52, 23.92]);
Map.centerObject(aoi, 11);

var bands = ['built', 'water', 'trees', 'grass', 'crops',
             'shrub_and_scrub', 'flooded_vegetation', 'bare'];

var dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
  .filterBounds(aoi)
  .filterDate('2026-01-01', '2026-03-31')
  .select(bands);
print('Dynamic World images used:', dw.size());

var lc = dw.mean().clip(aoi).toFloat();

Map.addLayer(lc.select('built'), {min: 0, max: 1, palette: ['ffffff', 'ff0000']}, 'Built-up probability');
Map.addLayer(lc.select('trees'), {min: 0, max: 1, palette: ['ffffff', '006400']}, 'Tree probability', false);

Export.image.toDrive({
  image: lc,
  description: 'Dhaka_DW_landcover_2026DrySeason',
  folder: 'HRRT_Dhaka',
  region: aoi,
  scale: 10,
  crs: 'EPSG:32646',
  maxPixels: 1e9
});
