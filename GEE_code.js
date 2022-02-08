// Defining a region of interest from a shape file prepared in ArcGIS
// and uploaded it to GEE as an asset
var aoi = ee.FeatureCollection('users/angelinkatula/AOI_');
aoi = aoi.geometry();
Map.centerObject(aoi);
Map.addLayer(aoi, {color: 'red'}, 'area of interest');

//Finding a picture that was preliminary chosen in USGS Earth Explorer
var Landssat8_2020 = ee.Image('LANDSAT/LC08/C01/T1_SR/LC08_196021_20200615')
.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
        ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']);
print(Landssat8_2020);


//Cropping the image to the area of interest
var L8image2020 = Landssat8_2020.clip(aoi);

//Compute NDVI (NIR-RED)/(NIR+RED) (to enhance vegetation)
var NDVI = L8image2020.normalizedDifference(['NIR', 'Red']).rename('NDVI');


//Compute NDBI (SWIR-NIR)/(SWIR+NIR) (to enhance build-up area)
var NDBI = L8image2020.normalizedDifference(['SWIR1', 'NIR']).rename('NDBI');


//Compute MNDWI (GREEN-SWIR)/(GREEN+SWIR) (to enhance open water)
var MNDWI = L8image2020.normalizedDifference(['Green', 'SWIR1']).rename('MNDWI');

//Adding calculated indices to our band stack
var image = L8image2020.addBands(NDVI).addBands(NDBI).addBands(MNDWI);
print(image);

// Generate a histogram  (can be done for any band)
var histogram = ui.Chart.image.histogram({
  image: image.select('Blue'),
  region: aoi,
  scale: 30
});
histogram.setOptions({
  title: 'Histogram of Blue band values'
});

print(histogram);



//Scaling band values to get a uniform scale from 0 to 1
var minMax = image.reduceRegion({
  reducer: ee.Reducer.minMax(),
  geometry: aoi,
  scale: 30,
  maxPixels: 10e9
});


// use unit scale to scale the pixel values, so that they have uniform scale from 0 to 1
var image_scaled = ee.ImageCollection.fromImages(
  image.bandNames().map(function(name){
    name = ee.String(name);
    var band = image.select(name);
    return band.unitScale(ee.Number(minMax.get(name.cat('_min'))), ee.Number(minMax.get(name.cat('_max'))))
})).toBands().rename(image.bandNames());



//Create variables for classes
var forest = ee.FeatureCollection('users/angelinkatula/forest_2020');
var agriculture = ee.FeatureCollection('users/angelinkatula/agriculture_2020');
var water = ee.FeatureCollection('users/angelinkatula/water_2020');
var artificial = ee.FeatureCollection('users/angelinkatula/artificial_2020');


//Merge them to create a training dataset
var trainingFeatures = forest.merge(agriculture).merge(water).merge(artificial);
print(trainingFeatures);

//Bands to use
var bands = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDBI', 'MNDWI'];

// Get the values for all pixels in each polygon in the training.
var training = image_scaled.sampleRegions({
  // Get the sample from the polygons FeatureCollection.
  collection: trainingFeatures,
  // Keep this list of properties from the polygons.
  properties: ['class'],
  // Set the scale to get Landsat pixels in the polygons.
  scale: 30,
});


//Unsupervised classification to identify p
//spectral clusters without any training data to get an idea which natural classes there are
// Instantiate the clusterer and train it.
var clusterer = ee.Clusterer.wekaKMeans(4).train(training);

// Cluster the input using the trained clusterer.
var result_clustered = image_scaled.cluster(clusterer);

// Display the clusters with random colors.
Map.addLayer(result_clustered.randomVisualizer(), {}, 'clusters');




// Make a Random Forest classifier and train it.
var rf_classifier = ee.Classifier.smileRandomForest(300);

// Train the classifier.
var trained_rf = rf_classifier.train(training, 'class', bands);


// Classify the input imagery.
var classified_rf = image_scaled.classify(trained_rf);

Map.addLayer(image, {bands: ['Red', 'Green', 'Blue']}, 'true color');
Map.addLayer(trainingFeatures, {}, 'training polygons');
Map.addLayer(classified_rf,
             {palette: ['yellow', 'grey', 'green', 'blue'], min: 0, max: 3},
             'land cover rf');

var trainAccuracy_rf = trained_rf.confusionMatrix();
print('Resubstitution error matrix: ', trainAccuracy_rf);
print('Training overall accuracy: ', trainAccuracy_rf.accuracy());



//Support vector machine
// Create an SVM classifier with custom parameters.
var svm_classifier = ee.Classifier.libsvm({
  kernelType: 'linear',
  //gamma: 1,
  cost: 10
});

// Train the classifier.
var trained_svm = svm_classifier.train(training, 'class', bands);

// Classify the image.
var classified_svm = image_scaled.classify(trained_svm);
Map.addLayer(classified_svm,
             {palette: ['yellow', 'grey', 'green', 'blue'], min: 0, max: 3},
             'land cover svm');
var trainAccuracy_svm = trained_svm.confusionMatrix();
print('Resubstitution error matrix: ', trainAccuracy_svm);
print('Training overall accuracy: ', trainAccuracy_svm.accuracy());


Export.image.toDrive({
  image: classified_rf,
  description: 'cassified_rf_2020',
  scale: 30,
  region: aoi
});

Export.image.toDrive({
  image: classified_svm,
  description: 'cassified_svm_2020',
  scale: 30,
  region: aoi
});

Export.image.toDrive({
  image: result_clustered,
  description: 'clusters_2020',
  scale: 30,
  region: aoi
});


//Accuracy assessment on independant validation dataset consisting of 50 points
var validation_data = ee.FeatureCollection('users/angelinkatula/validation_2020');
print(validation_data);


//Sample the input imagery to get Feature Collection of training data
var validation = image_scaled.sampleRegions(validation_data, ['class'], 30);

// Classify the validation data with the trained classifier (ranfom forest).
var validated_rf = validation.classify(trained_rf);

// Get a confusion matrix representing expected accuracy (random forest)
var testAccuracy_rf = validated_rf.errorMatrix('class', 'classification');
print('Validation error matrix rf: ', testAccuracy_rf);
print('Validation overall accuracy rf: ', testAccuracy_rf.accuracy());

var rf = ee.FeatureCollection([ee.Feature(null, {'Accuracy': testAccuracy_rf.accuracy(), 'Producer Accuracy':testAccuracy_rf.producersAccuracy(), 'User Accuracy':testAccuracy_rf.consumersAccuracy(), 'Kappa': testAccuracy_rf.kappa(), 'Error Matrix':testAccuracy_rf.array()})]);
Export.table.toDrive({collection: rf, description: 'accuracy_rf_2020',
 fileNamePrefix: 'accuracy_rf_2020', selectors: ['User Accuracy', 'Producer Accuracy', 'Accuracy','Kappa', 'Error Matrix']});

// Classify the validation data with the trained classifier (support vector machine).
var validated_svm = validation.classify(trained_svm);

// Get a confusion matrix representing expected accuracy (support vector machine)
var testAccuracy_svm = validated_svm.errorMatrix('class', 'classification');
print('Validation error matrix svm: ', testAccuracy_svm);
print('Validation overall accuracy svm: ', testAccuracy_svm.accuracy());

var svm = ee.FeatureCollection([ee.Feature(null, {'Accuracy': testAccuracy_svm.accuracy(), 'Producer Accuracy':testAccuracy_svm.producersAccuracy(), 'User Accuracy':testAccuracy_svm.consumersAccuracy(), 'Kappa': testAccuracy_svm.kappa(), 'Error Matrix':testAccuracy_svm.array()})]);
Export.table.toDrive({collection: svm, description: 'accuracy_svm_2020',
 fileNamePrefix: 'accuracy_svm_2020', selectors: ['User Accuracy', 'Producer Accuracy', 'Accuracy','Kappa', 'Error Matrix']});
