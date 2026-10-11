// ============================================================
// HRRT Dhaka | STEP 2: Current water/canal map + cross-checks
//                      + blind validation sample (for photo-interpretation)
// ============================================================

var aoi = ee.Geometry.Rectangle([90.28, 23.62, 90.52, 23.92]);
Map.centerObject(aoi, 12);

// ---- 1. Sentinel-2 dry-season composite and MNDWI (same as Step 1)
function maskS2(img) {
  var scl = img.select('SCL');
  var good = scl.neq(1).and(scl.neq(3)).and(scl.neq(8))
                .and(scl.neq(9)).and(scl.neq(10));
  return img.updateMask(good);
}
var s2med = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filterDate('2026-01-01', '2026-03-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(maskS2).median().clip(aoi);
var mndwi = s2med.normalizedDifference(['B3', 'B11']).rename('MNDWI');

// ---- 2. Otsu threshold (objective threshold, official GEE method)
function otsu(histogram) {
  histogram = ee.Dictionary(histogram);
  var counts = ee.Array(histogram.get('histogram'));
  var means = ee.Array(histogram.get('bucketMeans'));
  var size = means.length().get([0]);
  var total = counts.reduce(ee.Reducer.sum(), [0]).get([0]);
  var sum = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0]);
  var mean = sum.divide(total);
  var indices = ee.List.sequence(1, size);
  var bss = indices.map(function (i) {
    var aCounts = counts.slice(0, 0, i);
    var aCount = aCounts.reduce(ee.Reducer.sum(), [0]).get([0]);
    var aMeans = means.slice(0, 0, i);
    var aMean = aMeans.multiply(aCounts)
      .reduce(ee.Reducer.sum(), [0]).get([0]).divide(aCount);
    var bCount = total.subtract(aCount);
    var bMean = sum.subtract(aCount.multiply(aMean)).divide(bCount);
    return aCount.multiply(aMean.subtract(mean).pow(2)).add(
      bCount.multiply(bMean.subtract(mean).pow(2)));
  });
  return means.sort(bss).get([-1]);
}
var hist = mndwi.reduceRegion({
  reducer: ee.Reducer.histogram(255, 0.002),
  geometry: aoi, scale: 20, maxPixels: 1e9, bestEffort: true
}).get('MNDWI');
print('Otsu threshold on MNDWI:', otsu(hist));

// ---- 3. Three candidate water maps
var waterMNDWI = mndwi.gt(0).unmask(0).rename('class').clip(aoi);      // threshold 0
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(aoi).filterDate('2026-01-01', '2026-03-31')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .select('VV');
var waterS1 = s1.median().focalMedian(30, 'circle', 'meters')
  .lt(-15).unmask(0).clip(aoi);                                        // VV < -15 dB
var gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').unmask(0);
var waterJRC = gsw.gte(50).clip(aoi);                                  // persistent water

Map.addLayer(s2med, {bands: ['B4', 'B3', 'B2'], min: 0, max: 2500}, 'S2 true colour');
Map.addLayer(waterMNDWI.selfMask(), {palette: ['0000ff']}, 'Water: MNDWI > 0');
Map.addLayer(waterS1.selfMask(), {palette: ['ff0000']}, 'Water: Sentinel-1 VV < -15 dB', false);
Map.addLayer(waterJRC.selfMask(), {palette: ['ffff00']}, 'Water: JRC occurrence >= 50%', false);

// ---- 4. Water area (km2) and agreement between the three maps
function areaKm2(img) {
  return ee.Number(img.multiply(ee.Image.pixelArea()).divide(1e6)
    .reduceRegion({reducer: ee.Reducer.sum(), geometry: aoi, scale: 10,
                   maxPixels: 1e10, bestEffort: true}).values().get(0));
}
print('Water area km2 | MNDWI:', areaKm2(waterMNDWI));
print('Water area km2 | Sentinel-1:', areaKm2(waterS1));
print('Water area km2 | JRC:', areaKm2(waterJRC));
print('Overlap MNDWI & S1 (km2):', areaKm2(waterMNDWI.and(waterS1)));
print('Overlap MNDWI & JRC (km2):', areaKm2(waterMNDWI.and(waterJRC)));

// ---- 5. Stratified random sample: 60 mapped-water + 60 mapped-non-water points
var sample = waterMNDWI.stratifiedSample({
  numPoints: 0, classBand: 'class', region: aoi, scale: 10,
  classValues: [0, 1], classPoints: [60, 60], seed: 42, geometries: true
});

// BLIND version: map class is removed and the order shuffled, so labelling
// is done without knowing what the map says (avoids circular bias).
var pts = sample.randomColumn('rnd', 7).sort('rnd');
var n = pts.size();
var list = pts.toList(n);
var blind = ee.FeatureCollection(ee.List.sequence(1, n).map(function (i) {
  var f = ee.Feature(list.get(ee.Number(i).subtract(1)));
  var c = f.geometry().coordinates();
  return ee.Feature(f.geometry(), {id: i, lon: c.get(0), lat: c.get(1)});
}));
print('Validation points:', n);
Map.addLayer(blind, {color: 'ff00ff'}, 'Validation points', false);

Export.table.toDrive({
  collection: blind,
  description: 'validation_points_blind',
  folder: 'HRRT_Dhaka',
  fileFormat: 'CSV',
  selectors: ['id', 'lon', 'lat']
});
