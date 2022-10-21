import json
test = open('datasets/carla1/carla_panoptic/annotations/panoptic_train.json')
carla = json.load(test)
print(carla["images"])
