# PyTorchANN
Basic regression model, built using PyTorch

![alt text](https://github.com/VBukowsky81/PyTorchANN/blob/main/PyTorchANNPic.jpg)

Hello and good day, everyone!

Here I present a quick PyTorch-built regression ANN model. Nothing fancy. But I have shown similar example with Keras before, well, this time it's PyTorch.

PyTorch does is significantly differently. As you can see, it's written as a class, and then data feeding, and epochs are run by a loop.

It is obvious why PyTorch is using this format - enterprise-scale development. All the big companies use PyTorch like Tesla, Facebook, almost all. And you can see why - everything is neatly broken into sections - ANN build is here, activation functions are separately, forward pass is built by hand, then we loop through our data, and backprop.

So much better format for team efforts, and huge-size models, and generally larger scale development.

Anyway, all the same stuff as before is here. Data preprocessing, then class, constructor, layer setups, forward passes. Then the loop feeds it all, and does backprop calculations.

This time model is predictor of diabetes rates, among given patient data set. Nothing fancy, typical regression task.

Next, I will demonstrate CNN's.

Good day!

Victor
