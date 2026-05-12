[S5a] Building sentence-level training data ...
[S5a] Training on 500 conversation-turns, 20 epochs ...
  Epoch   1/20  train_loss=0.2588  val_loss=0.2390
  Epoch   2/20  train_loss=0.2244  val_loss=0.2314
  Epoch   3/20  train_loss=0.3222  val_loss=0.2690
---------------------------------------------------------------------------
AcceleratorError                          Traceback (most recent call last)
/tmp/ipykernel_2485/2780963768.py in <cell line: 0>()
      4 os.makedirs("models", exist_ok=True)
      5 
----> 6 history_s5a = train_sentence_model(
      7     train_samples=train_samples,
      8     val_samples=val_samples,

3 frames
/content/CENG467_Final/src/pruning/lnn_trainer.py in train_sentence_model(train_samples, val_samples, epochs, lr, turns_per_conv, decay_factor, device, save_path, encoder)
    132 
    133             optimizer.zero_grad()
--> 134             loss.backward()
    135             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    136             optimizer.step()

/usr/local/lib/python3.12/dist-packages/torch/_tensor.py in backward(self, gradient, retain_graph, create_graph, inputs)
    628                 inputs=inputs,
    629             )
--> 630         torch.autograd.backward(
    631             self, gradient, retain_graph, create_graph, inputs=inputs
    632         )


/usr/local/lib/python3.12/dist-packages/torch/autograd/__init__.py in backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)
    362     # some Python versions print out the first line of a multi-line function
    363     # calls in the traceback and some print out the last line
--> 364     _engine_run_backward(
    365         tensors,
    366         grad_tensors_,

/usr/local/lib/python3.12/dist-packages/torch/autograd/graph.py in _engine_run_backward(t_outputs, *args, **kwargs)
    863         unregister_hooks = _register_logging_hooks_on_whole_graph(t_outputs)
    864     try:
--> 865         return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
    866             t_outputs, *args, **kwargs
    867         )  # Calls into the C++ engine to run the backward pass

AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.