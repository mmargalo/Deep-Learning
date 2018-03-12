import sys
sys.path.insert(0, '../../tools/keras')
from DatasetGenerator import DatasetGenerator
from ImageTransform import ImageTransform

pathData = '../data/leedsbutterfly/images'
pathDataset = '../data/dataset.txt'
dataSize = 86
imgWidth = 300
imgHeight = 300


transformList = []
transformList.append(ImageTransform(0, [imgWidth, imgHeight]))
transformList.append(ImageTransform(10))

dataset = DatasetGenerator(pathData, pathDataset, transformList, imgWidth, imgHeight, dataSize, 'true')
print(dataset.getSize())
(x_train, x_label) = dataset.generate()
print(x_train)

