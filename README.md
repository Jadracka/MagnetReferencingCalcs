# MagnetReferencingCalcs
## Description

The `fit_circle_3D` function performs a fitting process to find a 3D circle that best represents a set of 3D points. The process involves the following steps:

1. **Fitting a Plane**:  
   Initially, a plane is fitted to the given 3D points using the `fit_plane` function. This plane represents the best linear fit to the data.

2. **Point Projection**:  
   The points are then projected onto the fitted plane. The projection of a point \((X, Y, Z)\) onto a plane with coefficients \((a, b, c, d)\) can be calculated as follows:

   \[
   x_{\text{proj}} = X - \frac{a(X + bY + cZ + d)}{a^2 + b^2 + c^2}
   \]

   \[
   y_{\text{proj}} = Y - \frac{b(X + bY + cZ + d)}{a^2 + b^2 + c^2}
   \]

   \[
   z_{\text{proj}} = Z - \frac{c(X + bY + cZ + d)}{a^2 + b^2 + c^2}
   \]

3. **Rotation Transformation**:  
   A rotation transformation is applied to align the plane with the XY plane. This transformation ensures that the normal vector of the plane becomes parallel to the Z-axis. The rotation matrix \( R \) for aligning the plane's normal vector \((a, b, c)\) with the Z-axis \((0, 0, 1)\) can be computed using the function `rotation_matrix_from_vectors` and applied to each point.

4. **Point-to-Point Distance Check**:  
   The distances between the original and transformed points are compared using the `compare_distances` function. This check helps identify any discrepancies introduced during the transformation.

5. **Fitting a 2D Circle**:  
   After transformation, the 3D problem is transformed into a 2D circle fitting problem on the XY plane. The function `fit_circle_2d` is then used to fit a circle to the transformed points.

## Parameters

- **data_tuple**: A tuple containing a dictionary of 3D points and the associated file path.
- **output_unit**: A dictionary specifying the desired output units for distances.
- **point_transform_check_tolerance**: Tolerance for checking differences in point-to-point distances after transformation.
- **log_statistics** *(optional)*: If `True`, log statistics will be generated and stored.

## Returns

A dictionary containing various information about the fitted circle, including:
- Center
- Radius
- Circle's normal vector
- Circle's name
- Offsets
- Plane statistics
- Plane angles parameters
- Circle statistics
