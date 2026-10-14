// ============================================================
// HRRT Dhaka | STEP 6a: Export historical data for temporal indicators
//   (1) GHSL built-up surface, 1975-2020, 100 m   -> encroachment over time
//   (2) JRC Global Surface Water yearly history, 30 m -> water presence over time
// Both are free, global, and consistent across years.
// ============================================================

var aoi = ee.Geometry.Rectangle([90.28, 23.62, 90.52, 23.92]);
Map.centerObject(aoi, 11);

// ---- 1) GHSL built-up surface (m2 of built-up per 100 m cell), epochs 1975-2020
var ghsl = ee.ImageCollection('JRC/GHSL/P2023A/GHS_BUILT_S')
  .filterDate('1970-01-01', '2021-06-01')
  .sort('system:time_start')
  .select('built_surface');
print('GHSL epochs found (expect 10):', ghsl.size());
var ghslStack = ghsl.toBands().clip(aoi).toFloat();
print('GHSL band names:', ghslStack.bandNames());
Map.addLayer(ghsl.sort('system:time_start').first().clip(aoi),
  {min: 0, max: 10000, palette: ['ffffff', 'ff0000']}, 'GHSL built-up, first epoch', false);
Map.addLayer(ghsl.sort('system:time_start', false).first().clip(aoi),
  {min: 0, max: 10000, palette: ['ffffff', 'ff0000']}, 'GHSL built-up, last epoch');

Export.image.toDrive({
  image: ghslStack,
  description: 'Dhaka_GHSL_builtup_1975_2020',
  folder: 'HRRT_Dhaka',
  region: aoi,
  scale: 100,
  crs: 'EPSG:32646',
  maxPixels: 1e9
});

// ---- 2) JRC yearly water class: 0 no data, 1 not water, 2 seasonal, 3 permanent
var years = [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2021];
var gsw = ee.ImageCollection('JRC/GSW1_4/YearlyHistory');
var gswStack = ee.Image.cat(years.map(function (y) {
  return gsw.filter(ee.Filter.calendarRange(y, y, 'year')).first()
            .select('waterClass').rename('y' + y);
})).clip(aoi).toFloat();
print('JRC yearly band names:', gswStack.bandNames());
Map.addLayer(gswStack.select('y1990').gte(2).selfMask(), {palette: ['0000ff']}, 'Water 1990 (JRC)', false);
Map.addLayer(gswStack.select('y2021').gte(2).selfMask(), {palette: ['ff0000']}, 'Water 2021 (JRC)', false);

Export.image.toDrive({
  image: gswStack,
  description: 'Dhaka_JRC_yearly_water_1990_2021',
  folder: 'HRRT_Dhaka',
  region: aoi,
  scale: 30,
  crs: 'EPSG:32646',
  maxPixels: 1e9
});
