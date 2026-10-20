// ============================================================
// HRRT Dhaka | STEP 7c: Which administrative districts does the study window cover?
// Gives the numbers for the sentence that describes the study area administratively.
// Source: FAO GAUL 2015 (level 1 = division, level 2 = district).
// ============================================================

var aoi = ee.Geometry.Rectangle([90.28, 23.62, 90.52, 23.92]);
var aoiKm2 = ee.Number(aoi.area(1)).divide(1e6);
print('Study window area (km2):', aoiKm2);

var districts = ee.FeatureCollection('FAO/GAUL/2015/level2')
  .filter(ee.Filter.eq('ADM0_NAME', 'Bangladesh'))
  .filterBounds(aoi);

var table = districts.map(function (f) {
  var km2 = ee.Number(f.geometry().intersection(aoi, 1).area(1)).divide(1e6);
  return ee.Feature(null, {
    district: f.get('ADM2_NAME'),
    division: f.get('ADM1_NAME'),
    km2_in_window: km2.format('%.1f'),
    pct_of_window: km2.divide(aoiKm2).multiply(100).format('%.1f'),
    sort_key: km2
  });
}).sort('sort_key', false);

print('Districts intersecting the window (largest share first):',
      table.reduceColumns(ee.Reducer.toList(4), ['district', 'division', 'km2_in_window', 'pct_of_window'])
           .get('list'));

Map.centerObject(aoi, 10);
Map.addLayer(districts.style({color: '0000ff', fillColor: '00000000', width: 1}), {}, 'Districts');
Map.addLayer(aoi, {color: 'red'}, 'Study window');
