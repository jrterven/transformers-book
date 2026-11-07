# Chapter 21 exercises

Image and Video Understanding

Exercises 1-4 use hand calculations; 5-8 inspect NumPy helpers; 8-12
are optional extensions. All twelve have [complete solutions](../../solutions/ch21_vision/README.md)
in the written PDF, executed notebook, and notebook PDF.
Use helpers from `examples.ch21_vision.model`. Unless stated otherwise,
attention counts are per head, include ordered query-key self-pairs,
and exclude special tokens.

1. **Make a patch grid.** Divide a $12\times8$ RGB image into square patches of side four. Count the patch rows and columns and the flattened features per patch. Give the projection-matrix shape and the token shape after a projection to width five and one classification token. Explain which operation merely rearranges values and which can discard information.

2. **Change the patch size.** Use a $32\times32$ image. Compare patch sizes eight and four. Count patch tokens and entries in the full patch-only score matrix for each case. Explain what information a smaller patch can separate.

3. **Check an input convention.** One channel value is 153 on a scale from 0 to 255. Convert it to $[0,1]$, then subtract 0.5 and divide by 0.25. Compare with what happens if the scale conversion is skipped. Name two other preprocessing choices that should match the checkpoint.

4. **Count video connections.** A clip has four frames and nine patches per frame. Compute the joint count and the separate temporal and spatial counts, ignoring special tokens. Draw one temporal group and one spatial group. Trace a temporal-then-spatial route from frame 1, location 2, to frame 4, location 7, naming the intermediate stage.

5. **Check row-major patch order.** Make a single-channel image with `np.arange(24).reshape(4,6,1)`. Call `patchify` with patch size two. Inspect the first and last output rows. Map their values back to coordinates in the original image.

6. **Add a position to an identical patch.** Make two identical patch rows of length four. Project them with a weight array of shape $[4,2]$ of your choice. Call `patch_embeddings` without `cls`. Compare the default output with a call that supplies `positions` with rows $(0,0)$ and $(0,1)$. Explain why the two output rows differ in the second call and why list order alone does not supply location to full self-attention.

7. **Compare ordered and sorted features.** Use ordered feature rows $(1,0,0,0)$, $(0,0,1,0)$, $(0,2,0,0)$, and $(0,0,0,2)$ with labels $(0,1,0,1)$. Fit `train_linear_probe`. Then sort each row and train a new head with the same settings. Each fit returns `weight, bias`. Use the class with the largest entry in `features @ weight + bias` for each training row. Compare the predictions and explain what the sorted head can distinguish.

8. **Trade temporal for spatial tokens (optional).** Use `space_time_pairs` to compare clips with $(T,S)=(2,12)$, $(4,6)$, and $(6,4)$. All have 24 patch tokens. Compare the joint and divided counts and derive their ratio $1/T+1/S$. Explain why swapping $T$ and $S$ changes the two components while preserving their sum.

9. **Form tubelets (optional).** Use six RGB frames of size $8\times12$. Divide the clip into nonoverlapping regions of two frames and $4\times4$ pixels. Count the tokens and the flattened input width of a tubelet. Compare the token count with taking patches of the same size separately in every frame.

10. **Find a leaking split (optional).** A collection has ten original recordings, and each recording produces three overlapping clips. Design a split for evaluating a model on new recordings. Then explain how the split should change if several recordings come from one camera and the goal is to test new cameras.

11. **Test an action's direction (optional).** Choose an action where reversing the frames changes the label. Design a comparison between an image classifier that averages scores over frames and a model with temporal attention. Keep the sampled frames the same for both models, and specify how the temporal model receives order information. State the averaged-image baseline's accuracy ceiling on balanced opposite-label pairs. Specify how you would score each original and reversed pair.

12. **Compare two visual checkpoints (optional).** One model uses smaller patches and is trained at a higher image resolution. Assume both checkpoints support a common resolution. Compare them there using the same crop and score. Then design a second experiment that changes resolution for one checkpoint. State which claim each comparison supports and which counts and timings to report.
