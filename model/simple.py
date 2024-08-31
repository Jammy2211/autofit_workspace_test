from os import path

import autofit as af

"""
__Simple__
"""
model = af.Model(af.ex.Gaussian)

print(model.info)

graph = af.VisualiseGraph(model=model)
file = path.join("model", "visualize", "simple.html")
graph.save(path=file)


"""
__Collection__
"""
model = af.Collection(
    gaussian=af.Model(af.ex.Gaussian),
    exponential=af.Model(af.ex.Exponential),
)

print()
print(model.info)

graph = af.VisualiseGraph(model=model)
file = path.join("model", "visualize", "collection.html")
graph.save(path=file)

"""

__
"""
