# Initial dictionary
myData = {'apple':'1', 'banana':'2', 'house':'3', 'car':'4', 'hippopotamus':'5'}

# Create the container class
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

# Finally create the instance and bind the dictionary to it
k = Struct(**myData)
print(k.apple)