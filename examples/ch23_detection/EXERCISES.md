# Chapter 23 exercises

Object Detection and Segmentation

Exercises 1-4 use hand calculations, 5-8 inspect the NumPy helpers, and
9-11 extend the ideas to evaluation design. Exercise 6 is an optional
edge-case investigation. All eleven have [complete solutions](../../solutions/ch23_detection/README.md)
in the written PDF, executed notebook and notebook PDF.
Use `examples.ch23_detection.model`. Box coordinates are continuous
left, top, right, bottom edges; mask indices are (row, column).

1. **Compare two boxes.** Use boxes $(1,1,4,3)$ and $(2,0,5,2)$ in corner coordinates. Compute both areas, the overlap area, and IoU. Move the second box one unit right and repeat.

2. **Count shared pixels.** On a $3\times3$ grid, let mask A mark $(0,0)$, $(0,1)$, and $(1,1)$. Let mask B mark $(0,1)$, $(1,1)$, $(1,2)$, and $(2,2)$. Treat each pair as (row, column). Draw the masks and compute intersection and union counts.

3. **Make a one-to-one assignment.** Use cost rows $(2,1)$, $(3,5)$, and $(4,2)$ for three queries and two objects. Find every minimum-cost assignment using a distinct query for each object. Identify the unmatched query in each case. Compare with choosing a minimum independently in each column.

4. **Track precision and recall.** A detector returns four candidates whose outcomes in score order are false positive, true positive, true positive, false positive. There are three annotated objects. Compute precision and recall after each candidate. Explain what recall cannot reach with these predictions. Form the precision envelope and compute all-points AP, including the unattained recall region.

5. **Check touching edges.** Call `box_iou` on $(0,0,2,3)$ and $(2,0,4,3)$. Move the second box left by 0.25 and compare the result. Describe how the edge convention affects both cases.

6. **Inspect an empty union (optional).** Create two Boolean masks of shape $[3,3]$ with no foreground. Call `mask_iou`. Add a foreground pixel to one mask and call it again. Explain why the two cases need different interpretations.

7. **Inspect matching indices.** Give the cost matrix from Exercise 3 to `match_queries`. Compare its zero-based `(query, target)` pairs with your hand assignment. Add 4 to cost entry `[0, 1]` and solve the new matrix.

8. **Change the overlap threshold.** Use annotated boxes $(0,0,3,3)$ and $(5,0,8,3)$. The predictions are $(1,0,4,3)$, $(0,0,3,3)$, and $(5,0,8,3)$ with scores $(0.9,0.8,0.7)$. Treat all boxes as one class in one image. Call `precision_recall` with the annotations, predictions, and scores at thresholds 0.5 and 0.75. Compare the true-positive flags. Pass the returned precision and recall to `all_points_ap` and compare the two results.

9. **Compare two prompting protocols (optional).** Design a study of masks made from annotated boxes and from detector boxes. Specify how to pair the examples, how to select one returned mask, and how to report missed detections. Explain which stage the first protocol isolates. For two annotated objects with one detected mask of IoU 0.8 and one missed object, compare detected-only and all-instance means and state how false positives are reported.

10. **Audit resized annotations (optional).** An image is resized and padded before inference. Describe the steps needed to compare predicted boxes and masks with annotations in the original image. Use an original image of width 4 and height 3, a scale of two, left padding 1, and top padding 2. Map box $(1,0,3,2)$ to the canvas and back. Compare the correct inverse with division by scale alone. Add a mask check that tests pixel values as well as shape.

11. **Separate detection errors (optional).** Choose four error categories from the chapter. Sketch a small image and a set of predictions that would demonstrate each category. Say what changing the confidence cutoff could fix. Which problems would still require a different prediction?
