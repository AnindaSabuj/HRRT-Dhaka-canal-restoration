// ============================================================
// HRRT Dhaka | STEP 1: Study area + free data loading + first look
// Run at: https://code.earthengine.google.com
// ============================================================

// ---- 1. Study area (box covering Dhaka city + Buriganga, Turag, Balu, Shitalakshya reaches)
var aoi = ee.Geometry.Rectangle([90.28, 23.62, 90.52, 23.92]);
Map.centerObject(aoi, 11);
Map.addLayer(aoi, {color: 'red'}, 'AOI', false);

// ---- 2. Sentinel-2 L2A, dry season Jan-Mar 2026 (least cloud, clearest canals)
function maskS2(img) {
  var scl = img.select('SCL');
  // remove saturated(1), cloud shadow(3), cloud medium(8), cloud high(9), cirrus(10)
  var good = scl.neq(1).and(scl.neq(3)).and(scl.neq(8))
                .and(scl.neq(9)).and(scl.neq(10));
  return img.updateMask(good);
}

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filterDate('2026-01-01', '2026-03-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(maskS2);
print('Sentinel-2 scenes used:', s2.size());

var s2med = s2.median().clip(aoi);
Map.addLayer(s2med, {bands: ['B4', 'B3', 'B2'], min: 0, max: 2500}, 'S2 true colour');

// ---- 3. MNDWI = (Green - SWIR1) / (Green + SWIR1)  -> Sentinel-2 B3 and B11
var mndwi = s2med.normalizedDifference(['B3', 'B11']).rename('MNDWI');
Map.addLayer(mndwi, {min: -0.8, max: 0.8, palette: ['8c510a', 'ffffff', '2166ac']}, 'MNDWI');

// First-pass water mask (threshold 0 is a starting point; refined in Step 2)
var water0 = mndwi.gt(0).selfMask();
Map.addLayer(water0, {palette: ['0000ff']}, 'Water (MNDWI > 0)');

// ---- 4. Sentinel-1 GRD (VV, VH) dry-season median, for SAR water check
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(aoi)
  .filterDate('2026-01-01', '2026-03-31')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .select(['VV', 'VH']);
print('Sentinel-1 scenes used:', s1.size());
var s1med = s1.median().clip(aoi);
Map.addLayer(s1med.select('VV'), {min: -25, max: 0}, 'S1 VV (dB)', false);

// ---- 5. Terrain: Copernicus GLO-30 DEM (free alternative to ALOS 12.5 m)
var dem = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').mosaic().clip(aoi);
Map.addLayer(dem, {min: 0, max: 20, palette: ['000080', '00ffff', '00ff00', 'ffff00', 'ff0000']}, 'DEM (m)', false);

// ---- 6. Independent reference: JRC Global Surface Water (1984-2021)
var gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').clip(aoi);
Map.addLayer(gsw.select('occurrence'),
  {min: 0, max: 100, palette: ['ffffff', 'ffbbbb', '0000ff']}, 'JRC water occurrence %', false);

// ---- 7. Landsat archive check: how many scenes exist per sensor (for change analysis later)
var l5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').filterBounds(aoi).filterDate('1990-01-01', '2000-12-31');
var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(aoi).filterDate('2021-01-01', '2026-03-31');
print('Landsat 5 scenes 1990-2000:', l5.size());
print('Landsat 8 scenes 2021-2026:', l8.size());

// ---- 8. Export first water map to Drive so you can open it in QGIS (optional)
Export.image.toDrive({
  image: mndwi.toFloat(),
  description: 'Dhaka_MNDWI_2026DrySeason',
  folder: 'HRRT_Dhaka',
  region: aoi,
  scale: 10,
  crs: 'EPSG:32646',   // WGS 84 / UTM zone 46N
  maxPixels: 1e9
});
