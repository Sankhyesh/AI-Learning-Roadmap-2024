### Understanding Padding and Strides in Convolutional Neural Networks

The material discusses two important operations in Convolutional Neural Networks (CNNs) that complement the convolution operation: **Padding** and **Strides**. These techniques help manage the spatial dimensions of feature maps and influence the network's computational efficiency and feature extraction capabilities.

#### I. Padding

- **The Problem with Standard Convolution:**
  ![alt text](images/43/
  image-1.png)
  _ **Shrinking Output Volume:** Each time a convolution operation is applied with a filter larger than 1x1 and no padding, the resulting **feature map** becomes smaller than the input volume. As illustrated in the material (similar to the scenario in 'image_08109f.png' where a 6x6 input might become a 4x4 output), if you have many convolution layers, the image dimensions can shrink rapidly, leading to significant **information loss**, especially from the deeper parts of the network. This is referred to as a "loss of resolution."
  _ **Underutilization of Border Pixels:** Pixels at the borders of an image are covered by the filter fewer times than pixels in the center. This means information from the edges of the image (which could be crucial) contributes less to the output and might be lost.

- **What is Padding?**
  ![alt text](images/43/
  image.png)
  _ **Padding** is the process of adding extra pixels (usually zeros) around the boundary of an input image or feature map before applying a convolution.
  _ The most common type is **zero-padding**, where all the added pixels have a value of zero. The material refers to this as "0-padding" in Keras. \* **Goal:** The primary goal of padding is often to preserve the spatial dimensions of the input so that the output feature map has the same height and width as the input. It also ensures that border pixels are treated more thoroughly by the filter.

- **How Padding Works & Its Effect on Output Size:**
![alt text](images/43/
image-2.png)
  - If an input image has dimensions $N \times N$ and the filter is $F \times F$, the output without padding is $(N - F + 1) \times (N - F + 1)$.
  - If we add $P$ layers of padding (e.g., $P=1$ means one layer of pixels added around the image), the effective size of the input image becomes $(N + 2P) \times (N + 2P)$.
  - The formula for the output dimension then becomes:
    $$\text{Output Size} = (N + 2P - F) + 1$$
    (This formula is shown being developed in 'image_081119.png', leading to $(N+2P-F)/S + 1$ when strides are included).
  - To ensure the output size is the same as the input size ($N \times N$) when stride is 1, we need $N + 2P - F + 1 = N$. This simplifies to $2P - F + 1 = 0$, or $P = (F-1)/2$. For example, if you have a $3 \times 3$ filter, you would need $P=(3-1)/2 = 1$ layer of zero-padding. If you have a $5 \times 5$ filter, you'd need $P=(5-1)/2 = 2$ layers of padding.
  - The image 'image_081119.png' demonstrates how a $5 \times 5$ image, when padded to become $7 \times 7$ (implying $P=1$), and then convolved with a $3 \times 3$ filter (with stride 1, though the image later uses stride 2 for a different example), would result in a $5 \times 5$ feature map if stride was 1: $(5 + 2(1) - 3) + 1 = 5$.

- **Padding in Keras:**

  - Keras provides two main options for the `padding` parameter in its convolutional layers (e.g., `Conv2D`):
    - **`padding='valid'`**: This means no padding is applied. The convolution is only computed where the filter fully overlaps with the input. This will result in the output size shrinking if $F > 1$.
    - **`padding='same'`**: Keras automatically adds zero-padding such that the output feature map has the **same** spatial dimensions as the input feature map (assuming a stride of 1). It calculates the required $P$ (usually $(F-1)/2$, potentially asymmetrically if needed).
  - **Code Example (from the material):**
    The material describes creating a sequential model in Keras:

    ```python
    # model_valid = Sequential()
    # model_valid.add(Conv2D(32, (3,3), padding='valid', input_shape=(28,28,1)))
    # # ... more layers with padding='valid'
    # model_valid.summary()
    ```

    With `padding='valid'`, an input of $28 \times 28$ convolved with a $3 \times 3$ filter becomes $26 \times 26$, then $24 \times 24$, and then $22 \times 22$ after three such layers.

    ```python
    # model_same = Sequential()
    # model_same.add(Conv2D(32, (3,3), padding='same', input_shape=(28,28,1)))
    # # ... more layers with padding='same'
    # model_same.summary()
    ```

    With `padding='same'`, the output dimensions remain $28 \times 28$ after each convolutional layer (assuming stride 1).

#### II. Strides

- **What are Strides?**
![alt text](images/43/
image-3.png)
  - The **stride** defines the step size the filter takes as it moves across the input image. It can be specified for horizontal and vertical movements independently.
  - A stride of $(1,1)$ means the filter moves one pixel at a time, both horizontally and vertically. This is the default in many cases.
  - A stride of $(S, S)$ means the filter jumps $S$ pixels at a time. The material refers to this as **strided convolution** when $S > 1$.
  - The image 'image_081119.png' visually depicts a stride of 1 (filter moving one step) and a stride of 2 (filter jumping two steps).

- **Effect of Strides on Output Size:**

  - Increasing the stride reduces the spatial dimensions of the output feature map.
  - The general formula for calculating the output size with input size $N$, filter size $F$, padding $P$, and stride $S$ is:
    $$\text{Output Size} = \lfloor \frac{N + 2P - F}{S} \rfloor + 1$$
    (This formula is highlighted in 'image_0810bc.png', 'image_081119.png', and 'image_081113.png'). The floor function ($\lfloor \cdot \rfloor$) is important.
  - **Handling Non-Divisible Cases (Floor Operation):**
  ![alt text](images/43/
  image-4.png)
    As shown in 'image_081113.png', if the term $(N + 2P - F)$ is not perfectly divisible by $S$, the result of the division is floored. For example, if $(N-F)/S = 1.5$ (assuming $P=0$), then $\lfloor 1.5 \rfloor = 1$. This means if the filter cannot make a full stride to fit another window, that partial window is ignored.
    - The example calculation in 'image_081113.png' for a $6 \times (\text{something})$ input, $3 \times 3$ filter, and stride $S=2$:
      Output rows = $\lfloor (6-3)/2 \rfloor + 1 = \lfloor 1.5 \rfloor + 1 = 1+1=2$.
    - Another example in 'image_081119.png' for a $7 \times 7$ input, $3 \times 3$ filter, stride $S=2$, no padding ($P=0$):
      Output size = $\lfloor (7-3)/2 \rfloor + 1 = \lfloor 4/2 \rfloor + 1 = 2+1=3$. So, a $3 \times 3$ feature map.
      If padding $P=1$ was applied (making input effectively $7+2(1)=9$ for that dimension calculation with respect to the formula $N+2P-F$):
      Output size = $\lfloor (7 + 2(1) - 3)/2 \rfloor + 1 = \lfloor (9-3)/2 \rfloor + 1 = \lfloor 6/2 \rfloor + 1 = 3+1=4$. So, a $4 \times 4$ feature map (as shown in one of the results in 'image_081119.png').

- **Why Use Strides (Reasons for Strided Convolutions)?**

  - **Downsampling / Reducing Spatial Dimensions:** Strides greater than 1 are an effective way to reduce the size of feature maps quickly within the network. This is sometimes used as an alternative to pooling layers.
  - **Capturing Higher-Level Features (Less Detail):** As mentioned in 'image_0810bc.png' ("High level features"), larger strides cause the filter to skip over parts of the input. This can lead the network to learn more coarse, **high-level features** rather than fine-grained, **low-level features**. This might be desirable if minute details are not critical for the task.
  - **Reducing Computational Cost & Memory:** Smaller feature maps mean fewer parameters in subsequent layers and less computation, which can speed up training and reduce memory usage. The material notes ("Computing ->" in 'image_0810bc.png') that this reason is less critical now with increased computing power, and strides of 1 are common.

- **Strides in Keras:**

  - The `strides` parameter in Keras `Conv2D` layers accepts a tuple, e.g., `strides=(S_h, S_v)`.
  - **Code Example (from the material):**

    ```python
    # model_strided = Sequential()
    # model_strided.add(Conv2D(32, (3,3), strides=(2,2), padding='valid', input_shape=(28,28,1)))
    # # ... more layers with strides=(2,2)
    # model_strided.summary()
    ```

    With an input of $28 \times 28$, no padding (`valid`), a $3 \times 3$ filter, and `strides=(2,2)`:
    Output size = $\lfloor (28 + 2(0) - 3)/2 \rfloor + 1 = \lfloor 25/2 \rfloor + 1 = \lfloor 12.5 \rfloor + 1 = 12 + 1 = 13$.
    The material shows the output becoming $14 \times 14$. Let's recheck the calculation provided in the image 'image_0810bc.png': $(28 + 2(0) - 3)/2 + 1 = (25/2)+1 = 12.5+1 = 13.5$. If Keras (or a specific interpretation) implies a different rounding or if the formula in the image $(N+2P-F)/S + 1$ always uses ceiling for _this specific library's "valid" calculation with stride_, or perhaps there's an implicit padding sometimes to make it work out to 14, it's worth noting. However, the standard formula with floor gives 13. The speaker's Keras output shows 14, then 7, then 4.

    - Let's assume Keras's `padding='valid'` might behave slightly differently or the formula presented $(N+2P-F+1)/S$ is used by the speaker, leading to $ (28-3+1)/2 = 26/2 = 13 $ if that were the case for size.
    - The handwritten calculation towards the end of the transcript for this Keras output: $N=28, P=0, F=3, S=2$.
      Formula used in note: $(\frac{N+2P-F}{S}) + 1 = (\frac{28 - 3}{2}) + 1 = \frac{25}{2} + 1 = 12.5 + 1 = 13.5$.
      The speaker says: "$28 + 2 - 3$ (this looks like $N+2P-F$ if $P=1$, but padding is 'valid' so $P=0$) ... $ / 2 ... 13.5 + 1 ... 13+1 (\text{floor}) = 14$".
            This calculation seems to apply floor to the $(N+2P-F)/S$ part _before_ adding 1 if the result is .5, rounding it up (ceiling-like for .5 specifically for Keras 'valid' output). For example, TensorFlow's `tf.nn.conv2d` with 'VALID' padding uses: `out_height = ceil((in_height - filter_height + 1) / stride_height)`. If $F=3, S=2$: `ceil((28-3+1)/2) = ceil(26/2) = 13`.
      If the Keras output is indeed 14, it suggests that for `padding='valid'` and stride > 1, the effective output size might be $\lceil N/S \rceil$ if the filter size is small, or a more complex formula specific to the library's implementation to align dimensions, or the speaker is using padding='same' in the actual code that produced 14 from 28 with stride 2.
      If `padding='same'` and `strides=2` for a $28 \times 28$ input, Keras aims for Output = $N/S = 28/2 = 14$.
      Let's assume the speaker's Keras output of 14 for the first layer with stride=2 implies `padding='same'` was actually used or Keras's 'valid' with stride has a specific behavior to get that result. Subsequent layers (14 to 7, 7 to 4) also follow this $N/S$ pattern, typical of `padding='same'`.
![alt text](images/43/
image-5.png)
    The key takeaway is that strides allow significant reduction in dimensionality.

#### III. Visualizing and Experimenting

- The material encourages hands-on practice: "try to visualize how padding and stride work on a 3D image (RGB image)." This is excellent advice for building intuition.
- The provided images ('image_081119.png', 'image_081113.png') are great for visualizing filter movement with strides and the effect on output size using the formulas.

#### IV. Key Takeaways & Connections

- **Padding** primarily addresses the issues of shrinking feature maps and underutilization of border pixels. It helps in **preserving information and spatial resolution**.
- **Strides** are used for **downsampling** the feature maps, which can lead to **computational savings** and encourage the learning of **higher-level features** by making the filter receptive field cover larger steps in the input.
- Both padding and strides are hyperparameters that need to be chosen carefully based on the specific CNN architecture and the problem being solved.
- Understanding the formulas for output size calculation is essential for designing CNN architectures and predicting the tensor shapes at each layer.

#### Stimulating Learning Prompts:

1.  How might the choice of stride value (e.g., 1 vs. 2 or more) affect the types of features a CNN learns in the early layers versus later layers?
2.  If you are designing a CNN for a task where precise localization of small objects is critical, would you prefer larger strides or smaller strides in your initial convolutional layers? Why? What role would padding play?
