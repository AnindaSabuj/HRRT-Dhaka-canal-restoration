// ============================================================
// HRRT Dhaka | STEP 7a: Export Bangladesh divisions (for the inset map)
// Source: FAO GAUL 2015, first-level administrative units (free).
// ============================================================

var divisions = ee.FeatureCollection('FAO/GAUL/2015/level1')
  .filter(ee.Filter.eq('ADM0_NAME', 'Bangladesh'))
  .map(function (f) { return f.simplify(500); });

print('Bangladesh divisions found (about 7 expected):', divisions.size());
print('Division names:', divisions.aggregate_array('ADM1_NAME'));

Map.centerObject(divisions, 6);
Map.addLayer(divisions.style({color: '0000ff', fillColor: '00000000', width: 1}), {}, 'Divisions');

// Study-area box used in every step (same as the AOI)
var aoi = ee.Geometry.Rectangle([90.28, 23.62, 90.52, 23.92]);
Map.addLayer(aoi, {color: 'red'}, 'Study area box');

Export.table.toDrive({
  collection: divisions.select(['ADM1_NAME']),
  description: 'Bangladesh_divisions',
  folder: 'HRRT_Dhaka',
  fileFormat: 'GeoJSON'
});
