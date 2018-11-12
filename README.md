# Current Work
#### DeepSupervised model performace
| single model |valid LB|public LB|
| ---------------- | ---- | ---|
|model_34 768*768|0.733|0.732|

#### Unet models only train data with ship
| model |valid LB| public LB (with empty list provide from DS model above) |
| ---------------- | ---- | ----|
|model_34 768*768|0.461|0.734|
|model_50 768*768|0.470|0.734|
|model_34+model_50 768*768||0.737|
|model_101 768*768|||

#### Post processing
remove small ships (less than 30 pixels) : +0.002 

## Tips from Heng
1. the metric is based on average of score  per image. hence you should to maximise score of each image;
2. the easiest image are those that contain no ships of few ships;
3. filter off small ship if you are not confidence of the results;
4. post process results of big ship (there is sometime small ship besdies big one);

## TODO List
1. loss function：Lsoftmax finetune？
2. model ensemble；
2. postprocessing：instance split + minAreaRect；
3. pseudo labeling：worth a try；











