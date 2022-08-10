
# Pytorch PlayGround

welcome,

Here I just share my codes of difference CNN architectures using pytorch framework

I'll be glad for you contributions

# Model Visualization
Here is a few tips to visualize your model:

- Using `Netron`: [[link]](https://netron.app/)

***Step 1:***

 you should import your model in `onnx` format:

To do so, create a dummy input and feed it into your network
```
x = torch.randn(50,1,32, 32)
model =Model()
input_names = ['Sentence']
output_names = ['yhat']
torch.onnx.export(model, x, 'rnn.onnx', input_names=input_names, output_names=output_names)
```

***Step 2:***
 
 Upload your model in `netron` website and see the layer-wise configuration of your network



